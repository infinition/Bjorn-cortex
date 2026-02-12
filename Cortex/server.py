"""
server.py - Bjorn AI Training Center Server
═══════════════════════════════════════════════════════════════════════════
Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

import os
import shutil
import asyncio
import json
import hashlib
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from starlette.requests import Request

# Security & MFA
import pyotp
import qrcode
import io
import base64
from jose import JWTError, jwt
import bcrypt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our trainer
from trainer import BjornModelTrainer

# --- Security Config ---
SECURITY_CONFIG_PATH = Path(__file__).parent / "security_config.json"
def default_security_config():
    return {
        "secret_key": "BJORN_DEFAULT_SECURE_KEY_8822",
        "algorithm": "HS256",
        "access_token_expire_minutes": 1440,
        "admin_password_hash": "$2b$12$9B8QzNwMUq4cvT.vSJz2xOOQv3v8S8GxQ3DckncehHk5fHTfE26ku",  # Default: admin
        "totp_secret": pyotp.random_base32(),
        "totp_setup_complete": False,
    }


def save_security_config():
    with open(SECURITY_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(SEC_CONFIG, f, indent=4)


def is_valid_totp_secret(secret: str) -> bool:
    if not isinstance(secret, str) or not secret.strip():
        return False
    try:
        # Forces base32 decode path used by pyotp without relying on current time.
        pyotp.TOTP(secret).at(0)
        return True
    except Exception:
        return False


def ensure_totp_secret_valid(config: dict) -> bool:
    current_secret = config.get("totp_secret", "")
    if is_valid_totp_secret(current_secret):
        return False
    config["totp_secret"] = pyotp.random_base32()
    config["totp_setup_complete"] = False
    return True


def load_security_config():
    if not SECURITY_CONFIG_PATH.exists():
        return default_security_config()
    try:
        with open(SECURITY_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ERROR] Invalid security config, using defaults: {e}")
        return default_security_config()

    defaults = default_security_config()
    defaults.update(cfg)
    ensure_totp_secret_valid(defaults)
    return defaults

SEC_CONFIG = load_security_config()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if ensure_totp_secret_valid(SEC_CONFIG):
        save_security_config()

    scheduler_task = asyncio.create_task(scheduler_loop())
    app.state.scheduler_task = scheduler_task
    try:
        yield
    finally:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Bjorn Cortex", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS (allow Pi to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
ICONS_DIR = BASE_DIR / "icons"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
ICONS_DIR.mkdir(exist_ok=True)

# Static files for icons
app.mount("/static/icons", StaticFiles(directory=str(ICONS_DIR)), name="icons")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Global Trainer Instance
trainer = BjornModelTrainer(base_dir=str(BASE_DIR))
training_in_progress = False

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self, history_file: Path):
        self.active_connections: List[WebSocket] = []
        self.log_history: List[Dict] = []
        self.current_session: List[Dict] = []
        self.past_sessions: List[Dict] = []
        self.history_file = history_file
        self.max_sessions = 10
        self.max_log_history = 100
        self._load_history()

    def _load_history(self):
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.past_sessions = data.get("past", [])
                    self.log_history = data.get("logs", [])
        except Exception as e:
            print(f"Failed to load history: {e}")

    def _save_history(self):
        try:
            with open(self.history_file, 'w') as f:
                json.dump({
                    "past": self.past_sessions,
                    "logs": self.log_history[-50:] # Keep last 50 logs
                }, f)
        except Exception as e:
            print(f"Failed to save history: {e}")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Store history
        msg_type = message.get("type")
        
        if msg_type == "log":
            self.log_history.append(message)
            if len(self.log_history) > self.max_log_history:
                self.log_history.pop(0)
        
        elif msg_type == "epoch_end":
            self.current_session.append(message)
        
        elif msg_type == "train_start":
            if self.current_session:
                version = "prev_" + datetime.now().strftime("%H%M%S")
                self.past_sessions.append({"id": len(self.past_sessions), "version": version, "data": self.current_session})
                if len(self.past_sessions) > self.max_sessions: self.past_sessions.pop(0)
                self._save_history()
            self.current_session = [] 
            
        elif msg_type == "train_result":
            if self.current_session:
                version = message.get("result", {}).get("version")
                if not any(s.get('version') == version for s in self.past_sessions):
                    self.past_sessions.append({"id": len(self.past_sessions), "version": version, "data": self.current_session})
                    if len(self.past_sessions) > self.max_sessions: self.past_sessions.pop(0)
                    self._save_history()
            
        # Broadcast to all connected clients
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"WS Send Error: {e}")
                pass

manager = ConnectionManager(BASE_DIR / "training_history.json")

# ═══════════════════════════════════════════════════════════════════════
# SECURITY LOGIC
# ═══════════════════════════════════════════════════════════════════════

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=SEC_CONFIG["access_token_expire_minutes"])
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SEC_CONFIG["secret_key"], algorithm=SEC_CONFIG["algorithm"])


def verify_admin_password(password: str) -> bool:
    stored_hash = SEC_CONFIG.get("admin_password_hash", "")
    password_bytes = password.encode("utf-8")

    # Preferred format: bcrypt hash ($2a/$2b/$2y)
    if stored_hash.startswith("$2"):
        try:
            return bcrypt.checkpw(password_bytes, stored_hash.encode("utf-8"))
        except ValueError:
            return False

    # Legacy compatibility: SHA-256 hex digest (64 chars)
    if len(stored_hash) == 64 and all(c in "0123456789abcdef" for c in stored_hash.lower()):
        is_match = hashlib.sha256(password_bytes).hexdigest() == stored_hash.lower()
        if is_match:
            # One-way migration to bcrypt once legacy hash is validated.
            SEC_CONFIG["admin_password_hash"] = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")
            save_security_config()
        return is_match

    return False

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token: raise HTTPException(status_code=401, detail="Missing Token")
    try:
        payload = jwt.decode(token, SEC_CONFIG["secret_key"], algorithms=[SEC_CONFIG["algorithm"]])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401, detail="Invalid Token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid Session")

@app.get("/setup", response_class=FileResponse)
async def setup_page(request: Request):
    if SEC_CONFIG.get("totp_setup_complete"):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("setup.html", {"request": request})

@app.get("/setup/qr")
async def get_setup_qr():
    if SEC_CONFIG.get("totp_setup_complete"):
        raise HTTPException(status_code=403, detail="Setup already complete")
    
    totp = pyotp.TOTP(SEC_CONFIG["totp_secret"])
    provisioning_url = totp.provisioning_uri(name="admin@BjornCortex", issuer_name="BjornCortex")
    
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buf = io.BytesIO()
    img.save(buf)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

@app.get("/setup/secret")
async def get_setup_secret():
    if SEC_CONFIG.get("totp_setup_complete"):
        raise HTTPException(status_code=403, detail="Setup already complete")
    return {"secret": SEC_CONFIG["totp_secret"]}

@app.get("/login", response_class=FileResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/token")
@limiter.limit("5/minute")
async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    # Check password
    if not verify_admin_password(form_data.password):
        await log_activity(request, f"FAILED LOGIN: {form_data.username}", "WARNING")
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    
    # Check TOTP (MFA)
    totp_code = request.headers.get("X-TOTP-Code")
    if not totp_code:
        return JSONResponse({"status": "mfa_required"}, status_code=202)
    
    if not is_valid_totp_secret(SEC_CONFIG.get("totp_secret", "")):
        await log_activity(request, "Invalid TOTP secret in security_config.json", "ERROR")
        raise HTTPException(status_code=503, detail="MFA non configure correctement")

    totp = pyotp.TOTP(SEC_CONFIG["totp_secret"])
    if not totp.verify(totp_code):
        await log_activity(request, f"FAILED MFA: {form_data.username}", "WARNING")
        raise HTTPException(status_code=401, detail="Code MFA invalide")

    # Lock setup after first successful MFA validation.
    if not SEC_CONFIG.get("totp_setup_complete"):
        SEC_CONFIG["totp_setup_complete"] = True
        save_security_config()
    
    access_token = create_access_token(data={"sub": form_data.username})
    await log_activity(request, f"LOGIN SUCCESS: {form_data.username}", "SUCCESS")
    return {"access_token": access_token, "token_type": "bearer"}

# ═══════════════════════════════════════════════════════════════════════
# HELPER: LOGGING
# ═══════════════════════════════════════════════════════════════════════

async def log_activity(request: Request, message: str, level: str = "INFO"):
    """Logs activity to console and broadcasts to UI."""
    client_ip = request.client.host if request.client else "unknown"
    print(f"[{level}] {message} (Client: {client_ip})")
    
    await manager.broadcast({
        "type": "log",
        "message": message,
        "level": level
    })

@app.get("/history")
async def get_history(user: str = Depends(get_current_user)):
    return {
        "logs": manager.log_history,
        "current": manager.current_session,
        "past": manager.past_sessions
    }

# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

CONFIG_FILE = BASE_DIR / "server_config.json"
SERVER_CONFIG = {
    "auto_train": True,
    "training_interval_minutes": 360,
    "use_tf_dataset": False,
    "default_training_epochs": 50,
    "training_batch_size": 32,
    "training_learning_rate": 0.001,
    "allow_device_api_without_auth": True,
    "device_api_key": "",
    "last_training_time": None,
    "next_training_time": None
}

def load_config():
    global SERVER_CONFIG
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                SERVER_CONFIG.update(json.load(f))
        except: pass

def save_config():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(SERVER_CONFIG, f, indent=4)
    except: pass

load_config()

def update_next_training_time():
    global SERVER_CONFIG
    if not SERVER_CONFIG["auto_train"]:
        SERVER_CONFIG["next_training_time"] = None
        return

    interval = SERVER_CONFIG["training_interval_minutes"]
    last = SERVER_CONFIG.get("last_training_time")
    
    if last:
        last_dt = datetime.fromisoformat(last)
        next_dt = last_dt.timestamp() + (interval * 60)
        SERVER_CONFIG["next_training_time"] = datetime.fromtimestamp(next_dt).isoformat()
    else:
        SERVER_CONFIG["next_training_time"] = datetime.now().isoformat()
    
    save_config()

@app.get("/config")
async def get_config(user: str = Depends(get_current_user)):
    if SERVER_CONFIG["auto_train"] and not SERVER_CONFIG["next_training_time"]:
        update_next_training_time()
    return SERVER_CONFIG

@app.post("/config/update")
async def update_config(request: Request, user: str = Depends(get_current_user)):
    global SERVER_CONFIG
    data = await request.json()
    for key in [
        "auto_train",
        "training_interval_minutes",
        "use_tf_dataset",
        "default_training_epochs",
        "training_batch_size",
        "training_learning_rate",
        "allow_device_api_without_auth",
        "device_api_key"
    ]:
        if key in data: SERVER_CONFIG[key] = data[key]

    # Minimal sanity constraints for runtime safety.
    SERVER_CONFIG["default_training_epochs"] = max(1, int(SERVER_CONFIG.get("default_training_epochs", 50)))
    SERVER_CONFIG["training_batch_size"] = max(1, int(SERVER_CONFIG.get("training_batch_size", 32)))
    SERVER_CONFIG["training_learning_rate"] = max(1e-6, float(SERVER_CONFIG.get("training_learning_rate", 0.001)))
            
    update_next_training_time()
    save_config()
    await log_activity(request, "Configuration Updated", "INFO")
    return SERVER_CONFIG


def _decode_bearer_user(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        return None
    try:
        payload = jwt.decode(token, SEC_CONFIG["secret_key"], algorithms=[SEC_CONFIG["algorithm"]])
        username = payload.get("sub")
        return username if username else None
    except JWTError:
        return None


def _is_device_allowed(request: Request) -> bool:
    if SERVER_CONFIG.get("allow_device_api_without_auth", True):
        return True
    configured_key = str(SERVER_CONFIG.get("device_api_key", "")).strip()
    if not configured_key:
        return False
    provided_key = request.headers.get("X-Device-Key", "")
    return secrets.compare_digest(provided_key, configured_key)


def authorize_user_or_device(request: Request) -> Tuple[str, str]:
    user = _decode_bearer_user(request)
    if user:
        return "user", user
    if _is_device_allowed(request):
        return "device", "device"
    raise HTTPException(status_code=401, detail="Missing Token")


@app.post("/upload")
async def upload_training_data(request: Request, file: UploadFile = File(...), mac_addr: str = None):
    authorize_user_or_device(request)
    try:
        client_ip = request.client.host if request.client else "unknown"
        prefix = "unknown"
        if mac_addr:
            prefix = "".join(c for c in mac_addr if c.isalnum())[-6:]
        else:
             prefix = client_ip.replace('.', '')[-6:]

        safe_filename = f"{prefix}_{file.filename}"
        file_path = DATA_DIR / safe_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        await log_activity(request, f"Received: {safe_filename} from {client_ip}", "SUCCESS")
        
        if SERVER_CONFIG.get("training_interval_minutes") == 0 and SERVER_CONFIG.get("auto_train"):
            await trigger_training_process()
            
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        await log_activity(request, f"Upload Error: {str(e)}", "ERROR")
        return {"status": "error", "message": str(e)}

@app.get("/model/latest")
async def get_latest_model_info(request: Request):
    authorize_user_or_device(request)
    files = sorted([f for f in MODELS_DIR.glob("bjorn_model_*.json") if "_weights.json" not in f.name])
    if not files: return {"version": None}
    
    latest = files[-1]
    with open(latest, 'r') as f: config = json.load(f)
    return config

@app.get("/model/download/{filename}")
async def download_model(request: Request, filename: str):
    authorize_user_or_device(request)
    file_path = MODELS_DIR / filename
    client_ip = request.client.host if request.client else "unknown"
    if file_path.exists():
        await log_activity(request, f"Downloading: {filename} from {client_ip}", "INFO")
        return FileResponse(file_path)
    
    await log_activity(request, f"Download Failed (404): {filename} (Client: {client_ip})", "ERROR")
    return JSONResponse(status_code=404, content={"message": "File not found"})

@app.get("/data/files")
async def list_data_files(user: str = Depends(get_current_user)):
    files = []
    for f in list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.csv.gz")):
        files.append({"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime})
    return sorted(files, key=lambda x: x["mtime"], reverse=True)

@app.delete("/data/files/{filename}")
async def delete_data_file(request: Request, filename: str, user: str = Depends(get_current_user)):
    file_path = DATA_DIR / filename
    if file_path.exists():
        try:
            file_path.unlink()
            await log_activity(request, f"Deleted file: {filename}", "WARNING")
            return {"status": "success"}
        except Exception as e:
            await log_activity(request, f"Delete Error: {str(e)}", "ERROR")
            return JSONResponse(status_code=500, content={"message": str(e)})
    return JSONResponse(status_code=404, content={"message": "File not found"})

@app.get("/stats")
async def get_stats(user: str = Depends(get_current_user)):
    data_files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.csv.gz"))
    models = [f for f in MODELS_DIR.glob("bjorn_model_*.json") if "_weights.json" not in f.name]
    return {
        "data_files_count": len(data_files),
        "models_count": len(models),
        "total_data_size_bytes": sum(f.stat().st_size for f in data_files)
    }

@app.post("/train/start")
async def start_training_manually(request: Request, epochs: Optional[int] = None, user: str = Depends(get_current_user)):
    if training_in_progress:
        return {"status": "already_running"}
    
    await trigger_training_process(epochs=epochs)
    await log_activity(request, "Manual training initiated", "INFO")
    return {"status": "started"}

# ═══════════════════════════════════════════════════════════════════════
# TRAINING PROCESS
# ═══════════════════════════════════════════════════════════════════════

async def trigger_training_process(epochs: Optional[int] = None):
    global training_in_progress
    if training_in_progress:
        return
    
    training_in_progress = True
    loop = asyncio.get_event_loop()
    
    # Callbacks for the Trainer thread to talk to Async WebSockets
    def thread_safe_status(data):
        asyncio.run_coroutine_threadsafe(manager.broadcast(data), loop)
        
    def thread_safe_log(msg):
        level = "ERROR" if "ERROR" in msg else ("SUCCESS" if "SUCCESS" in msg else "INFO")
        asyncio.run_coroutine_threadsafe(manager.broadcast({
            "type": "log", "message": msg, "level": level
        }), loop)

    trainer.set_log_callback(thread_safe_log)
    
    def run_training_and_broadcast():
        global training_in_progress
        try:
            use_stream = SERVER_CONFIG.get("use_tf_dataset", False)
            epochs_to_use = max(1, int(epochs if epochs is not None else SERVER_CONFIG.get("default_training_epochs", 50)))
            batch_size = max(1, int(SERVER_CONFIG.get("training_batch_size", 32)))
            learning_rate = max(1e-6, float(SERVER_CONFIG.get("training_learning_rate", 0.001)))
            result = trainer.train(
                thread_safe_status,
                epochs=epochs_to_use,
                use_stream=use_stream,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            if result:
                SERVER_CONFIG["last_training_time"] = datetime.now().isoformat()
                save_config()
                update_next_training_time()
                
                asyncio.run_coroutine_threadsafe(manager.broadcast({
                    "type": "train_result", "result": result
                }), loop)
        except Exception as e:
            thread_safe_log(f"Training Panic: {e}")
        finally:
            training_in_progress = False

    loop.run_in_executor(None, run_training_and_broadcast)

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Scheduler
async def scheduler_loop():
    while True:
        try:
            await asyncio.sleep(60)
            if not SERVER_CONFIG["auto_train"]: continue
            
            # If interval is 0, we use real-time upload triggers instead of background loop
            if SERVER_CONFIG["training_interval_minutes"] == 0:
                SERVER_CONFIG["next_training_time"] = None
                continue
                
            next_time = SERVER_CONFIG.get("next_training_time")
            if not next_time: update_next_training_time(); continue
                
            now = datetime.now()
            scheduled = datetime.fromisoformat(next_time)
            
            if now >= scheduled and not training_in_progress:
                data_files = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.csv.gz"))
                if len(data_files) > 0:
                    print(f"Auto-Training triggered at {now}")
                    await trigger_training_process(epochs=50)
                else:
                    SERVER_CONFIG["last_training_time"] = datetime.now().isoformat()
                    update_next_training_time()
        except Exception as e:
            print(f"Scheduler Error: {e}")
            await asyncio.sleep(60)

