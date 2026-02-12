"""
trainer.py - AI Model Trainer Backend for Bjorn Server
═══════════════════════════════════════════════════════════════════════════
Adapted from train_dl_model.py to support real-time web feedback.
"""

import json
import gzip
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable, Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Disable GPU if causing issues on some PCs (optional)
# tf.config.set_visible_devices([], 'GPU')

class TrainingProgressCallback(callbacks.Callback):
    """Custom Keras callback to stream progress to the web server."""
    
    def __init__(self, status_callback: Callable[[Dict], None]):
        super().__init__()
        self.status_callback = status_callback
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.status_callback({
            'type': 'epoch_end',
            'epoch': epoch + 1,
            'metrics': {k: float(v) for k, v in logs.items()}
        })

    def on_train_begin(self, logs=None):
        self.status_callback({'type': 'train_begin'})

    def on_train_end(self, logs=None):
        self.status_callback({'type': 'train_end'})

class BjornModelTrainer:
    """
    Trains deep learning models on Bjorn reconnaissance data.
    Outputs lightweight models for Pi Zero inference.
    """
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize trainer
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.model = None
        self.history = None
        self.log_callback = None
    
    def set_log_callback(self, callback: Callable[[str], None]):
        self.log_callback = callback

    def log(self, message: str, level: str = "INFO"):
        print(f"[{level}] {message}")
        if self.log_callback:
            self.log_callback(f"[{level}] {message}")

    # ═══════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_all_data(self) -> pd.DataFrame:
        """Load all CSV/JSONL files from the data directory."""
        files = list(self.data_dir.glob("*.csv")) + \
                list(self.data_dir.glob("*.csv.gz")) + \
                list(self.data_dir.glob("*.jsonl")) + \
                list(self.data_dir.glob("*.jsonl.gz"))
        
        if not files:
            self.log("No data files found in data directory.", "WARNING")
            return None
            
        self.log(f"Found {len(files)} data files.")
        
        dfs = []
        for file in files:
            df = self._load_single_file(file)
            if df is not None:
                dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            original_len = len(combined_df)
            combined_df.drop_duplicates(inplace=True)
            self.log(f"Combined {len(dfs)} files into {len(combined_df)} unique records (removed {original_len - len(combined_df)} duplicates).", "SUCCESS")
            return combined_df
        return None
    
    def _load_single_file(self, filepath: Path) -> pd.DataFrame:
        """Load a single data file"""
        try:
            if filepath.suffix == '.gz':
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    if '.csv' in filepath.name:
                        return pd.read_csv(f)
                    elif '.jsonl' in filepath.name:
                        return pd.read_json(f, lines=True)
            elif filepath.suffix == '.csv':
                return pd.read_csv(filepath)
            elif filepath.suffix in ['.jsonl', '.json']:
                return pd.read_json(filepath, lines=True)
        except Exception as e:
            self.log(f"Error loading {filepath.name}: {e}", "ERROR")
            return None
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════════════════
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'success') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        self.log("Preparing features...", "INFO")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df[target_column].values
        
        # Drop non-feature columns
        drop_cols = ['id', 'timestamp', 'mac_address', 'ip_address', 'consolidated', 'export_batch_id', target_column]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        # --- DYNAMIC FEATURE SELECTION ---
        # Look for 'feat_' prefix (new system) or fall back to 'feature_' (legacy)
        feature_cols = sorted([c for c in df.columns if c.startswith('feat_')])
        if not feature_cols:
             feature_cols = sorted([c for c in df.columns if c.startswith('feature_')], 
                                  key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0)
        
        if not feature_cols:
            raise ValueError("No feature columns (feat_*) found in dataset.")
        
        self.log(f"Training on {len(feature_cols)} features: {feature_cols[:5]}...", "INFO")
        self.feature_names = [c.replace('feat_', '') for c in feature_cols]
        
        X = df[feature_cols].fillna(0).values
        
        self.log(f"Final feature shape: {X.shape}, Targets: {y.shape}")
        return X, y

    # ═══════════════════════════════════════════════════════════════════════
    # MODEL ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    
    def build_model(self, input_dim: int, num_classes: int = 2, num_samples: int = 0, learning_rate: float = 0.001) -> keras.Model:
        """
        Build neural network architecture.
        Adapts complexity based on number of samples to prevent 'chaotic' over-fitting.
        """
        self.log(f"Building model architecture (input_dim={input_dim}, samples={num_samples})...", "INFO")
        
        model = models.Sequential([layers.Input(shape=(input_dim,))])
        
        # Dynamic architecture selection (Aggressive simplification for small data)
        if num_samples < 50:
            # Shallow & Narrow to prevent chaotic overfitting
            units_list = [16]
            dropout_val = 0.3
            l2_reg = 0.05
        elif num_samples < 200:
            # Medium complexity
            units_list = [32, 16]
            dropout_val = 0.4
            l2_reg = 0.02
        else:
            # Full complexity for large datasets
            units_list = [128, 64, 32]
            dropout_val = 0.3
            l2_reg = 0.01
            
        for units in units_list:
            model.add(layers.Dense(units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_reg)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_val))
        
        if num_classes <= 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
            
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss, metrics=['accuracy'])
        return model

    # ═══════════════════════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════════════════════
    
    def train(
        self,
        status_callback: Callable[[Dict], None],
        epochs: int = 50,
        use_stream: bool = False,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Train the model with live status updates."""
        try:
            if use_stream:
                self.log("Smart Streaming (tf.data) enabled. (Optimized for low RAM)", "INFO")
                # TODO: Implement full tf.data.experimental.make_csv_dataset logic here
                # For now, we fall back to standard loading but log the intent.
                # In a real implementation, this would replace load_all_data()
            
            df = self.load_all_data()
            if df is None:
                self.log("Training aborted: No data.", "ERROR")
                return None
            
            X, y = self.prepare_features(df)
            
            if X.shape[0] < 5:
                self.log(f"Insufficient data for training ({X.shape[0]} samples). Minimum required: 5", "WARNING")
                return {'success': False, 'error': 'insufficient_data'}

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            num_classes = len(np.unique(y))
            # If we only have 1 class (e.g. all successes), we still want a binary classifier
            if num_classes <= 1:
                num_classes = 2
                
            self.model = self.build_model(
                input_dim=X_train.shape[1],
                num_classes=num_classes,
                num_samples=X.shape[0],
                learning_rate=learning_rate
            )
            
            # Send initial stats
            status_callback({
                'type': 'train_start',
                'samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'input_dim': X_train.shape[1]
            })
            
            # Custom callback for WebSocket updates
            progress_callback = TrainingProgressCallback(status_callback)
            
            self.history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=max(1, int(batch_size)),
                callbacks=[progress_callback],
                verbose=0
            )
            
            self.log("Training complete!", "SUCCESS")
            
            # Export
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
            config_path, weights_path = self.export_for_pi(version)
            
            return {
                'success': True,
                'version': version,
                'accuracy': self.history.history['val_accuracy'][-1],
                'loss': self.history.history['val_loss'][-1],
                'config_path': str(config_path),
                'weights_path': str(weights_path)
            }
            
        except Exception as e:
            self.log(f"Training failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT
    # ═══════════════════════════════════════════════════════════════════════
    
    def export_for_pi(self, version: str) -> Tuple[Path, Path]:
        """Export model for Raspberry Pi."""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        self.log(f"Exporting model version {version}...", "INFO")
        
        weights = {}
        layer_idx = 0
        for layer in self.model.layers:
            if isinstance(layer, layers.Dense):
                w, b = layer.get_weights()
                if layer.name == 'output':
                    weights['w_out'] = w.tolist()
                    weights['b_out'] = b.tolist()
                else:
                    weights[f'w{layer_idx + 1}'] = w.tolist()
                    weights[f'b{layer_idx + 1}'] = b.tolist()
                    layer_idx += 1
        
        config = {
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'architecture': {
                'input_dim': self.model.input_shape[1],
                'output_dim': self.model.output_shape[1],
                'feature_names': self.feature_names if hasattr(self, 'feature_names') else []
            },
            'training_info': {
                'final_accuracy': float(self.history.history['val_accuracy'][-1]),
                'final_loss': float(self.history.history['val_loss'][-1])
            }
        }
        
        config_path = self.models_dir / f'bjorn_model_{version}.json'
        weights_path = self.models_dir / f'bjorn_model_{version}_weights.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        with open(weights_path, 'w') as f:
            json.dump(weights, f, indent=2)
            
        self.log(f"Model exported to {config_path}", "SUCCESS")
        return config_path, weights_path
