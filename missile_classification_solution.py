# Missile Flight Phase Classification - MLOps Solution
# Complete implementation with LSTM model, Docker deployment, and Kubernetes orchestration

import os
import json
import logging
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import joblib
from typing import Tuple, Dict, Any, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('missile_classification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles data loading, preprocessing, and feature engineering for missile tracking data.
    """
    
    def __init__(self, sequence_length: int = 10):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Length of sequences for LSTM input
        """
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['latitude', 'longitude', 'altitude', 'radiometric_intensity']
        
    def load_data(self, train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load training and test data from CSV files.
        
        Args:
            train_path: Path to training CSV file
            test_path: Path to test CSV file (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        
        test_df = None
        if test_path:
            logger.info(f"Loading test data from {test_path}")
            test_df = pd.read_csv(test_path)
            
        logger.info(f"Training data shape: {train_df.shape}")
        if test_df is not None:
            logger.info(f"Test data shape: {test_df.shape}")
            
        return train_df, test_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from raw sensor data.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Sort by track_id and timestamp for proper sequence creation
        df = df.sort_values(['track_id', 'timestamp'])
        
        # Calculate velocity components
        df['velocity_lat'] = df.groupby('track_id')['latitude'].diff().fillna(0)
        df['velocity_lon'] = df.groupby('track_id')['longitude'].diff().fillna(0)
        df['velocity_alt'] = df.groupby('track_id')['altitude'].diff().fillna(0)
        
        # Calculate speed and acceleration
        df['speed'] = np.sqrt(df['velocity_lat']**2 + df['velocity_lon']**2 + df['velocity_alt']**2)
        df['acceleration'] = df.groupby('track_id')['speed'].diff().fillna(0)
        
        # Calculate distance from origin
        df['distance_from_origin'] = np.sqrt(df['latitude']**2 + df['longitude']**2 + df['altitude']**2)
        
        # Rate of change of radiometric intensity
        df['radiometric_rate'] = df.groupby('track_id')['radiometric_intensity'].diff().fillna(0)
        
        # Update feature columns to include engineered features
        self.feature_columns = [
            'latitude', 'longitude', 'altitude', 'radiometric_intensity',
            'velocity_lat', 'velocity_lon', 'velocity_alt', 'speed', 
            'acceleration', 'distance_from_origin', 'radiometric_rate'
        ]
        
        logger.info(f"Engineered {len(self.feature_columns)} features")
        return df
    
    def create_sequences(self, df: pd.DataFrame, include_labels: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input from the dataframe.
        
        Args:
            df: Input dataframe
            include_labels: Whether to include labels in the output
            
        Returns:
            Tuple of (X_sequences, y_sequences) or (X_sequences, None)
        """
        X_sequences = []
        y_sequences = []
        
        # Group by track_id to create sequences within each track
        for track_id in df['track_id'].unique():
            track_data = df[df['track_id'] == track_id].copy()
            
            if len(track_data) < self.sequence_length:
                continue
                
            # Normalize features for this track
            track_features = track_data[self.feature_columns].values
            track_features_scaled = self.scaler.fit_transform(track_features)
            
            # Create sequences
            for i in range(len(track_features_scaled) - self.sequence_length + 1):
                X_sequences.append(track_features_scaled[i:i + self.sequence_length])
                
                if include_labels:
                    # Use the label of the last point in the sequence
                    y_sequences.append(track_data.iloc[i + self.sequence_length - 1]['reentry_phase'])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if include_labels else None
        
        logger.info(f"Created {len(X_sequences)} sequences of shape {X_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def preprocess(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting data preprocessing")
        
        # Engineer features
        train_df = self.engineer_features(train_df)
        if test_df is not None:
            test_df = self.engineer_features(test_df)
        
        # Handle missing values
        train_df = train_df.fillna(train_df.mean())
        if test_df is not None:
            test_df = test_df.fillna(test_df.mean())
        
        # Fit scaler on training data
        self.scaler.fit(train_df[self.feature_columns])
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_df, include_labels=True)
        
        X_test, y_test = None, None
        if test_df is not None:
            X_test, y_test = self.create_sequences(test_df, include_labels='reentry_phase' in test_df.columns)
        
        # Split training data into train/validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        result = {
            'X_train': X_train_split,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train_split,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'n_features': len(self.feature_columns)
        }
        
        logger.info("Preprocessing completed successfully")
        return result

class LSTMClassifier:
    """
    LSTM model for missile flight phase classification.
    """
    
    def __init__(self, sequence_length: int, n_features: int, lstm_units: int = 64):
        """
        Initialize the LSTM classifier.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of features per timestep
            lstm_units: Number of LSTM units
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.model = None
        self.history = None
        
    def build_model(self) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, 
                 input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(self.lstm_units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        logger.info("Starting model training")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.history.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input sequences
            batch_size: Batch size for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        start_time = time.time()
        predictions = self.model.predict(X, batch_size=batch_size)
        inference_time = time.time() - start_time
        
        logger.info(f"Inference completed in {inference_time:.4f} seconds for {len(X)} samples")
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input sequences
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        y_pred_binary = (predictions > 0.5).astype(int).flatten()
        
        # Calculate metrics
        report = classification_report(y, y_pred_binary, output_dict=True)
        auc_score = roc_auc_score(y, predictions)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'auc_roc': auc_score
        }
        
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

class BaselineClassifier:
    """
    Simple baseline classifier for comparison.
    """
    
    def __init__(self, method: str = 'altitude_threshold'):
        """
        Initialize baseline classifier.
        
        Args:
            method: Baseline method ('altitude_threshold', 'random_forest', 'dummy')
        """
        self.method = method
        self.model = None
        self.threshold = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the baseline classifier.
        
        Args:
            X: Training data
            y: Training labels
        """
        if self.method == 'altitude_threshold':
            # Use altitude feature (index 2) to determine threshold
            altitudes = X[:, -1, 2]  # Last timestep, altitude feature
            reentry_altitudes = altitudes[y == 1]
            non_reentry_altitudes = altitudes[y == 0]
            
            # Find optimal threshold
            thresholds = np.linspace(altitudes.min(), altitudes.max(), 100)
            best_accuracy = 0
            best_threshold = thresholds[0]
            
            for threshold in thresholds:
                predictions = (altitudes < threshold).astype(int)
                accuracy = np.mean(predictions == y)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            self.threshold = best_threshold
            logger.info(f"Optimal altitude threshold: {self.threshold:.2f}")
            
        elif self.method == 'random_forest':
            # Flatten sequences for random forest
            X_flat = X.reshape(X.shape[0], -1)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_flat, y)
            
        elif self.method == 'dummy':
            self.model = DummyClassifier(strategy='most_frequent')
            X_flat = X.reshape(X.shape[0], -1)
            self.model.fit(X_flat, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using baseline method.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        if self.method == 'altitude_threshold':
            altitudes = X[:, -1, 2]
            return (altitudes < self.threshold).astype(float)
        
        elif self.method in ['random_forest', 'dummy']:
            X_flat = X.reshape(X.shape[0], -1)
            return self.model.predict_proba(X_flat)[:, 1]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate baseline performance.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        y_pred_binary = (predictions > 0.5).astype(int)
        
        report = classification_report(y, y_pred_binary, output_dict=True)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'] if '1' in report else 0.0,
            'recall': report['1']['recall'] if '1' in report else 0.0,
            'f1_score': report['1']['f1-score'] if '1' in report else 0.0,
        }
        
        if len(np.unique(predictions)) > 1:
            metrics['auc_roc'] = roc_auc_score(y, predictions)
        else:
            metrics['auc_roc'] = 0.5
        
        logger.info(f"Baseline ({self.method}) metrics: {metrics}")
        return metrics

class ModelInferenceService:
    """
    Production inference service with logging and monitoring.
    """
    
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initialize inference service.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.load_components()
        
    def load_components(self):
        """Load model and preprocessing components."""
        try:
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise
    
    def preprocess_input(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for inference.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Preprocessed numpy array
        """
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        
        # Apply same feature engineering as training
        preprocessor = DataPreprocessor()
        df = preprocessor.engineer_features(df)
        
        # Scale features
        features = df[preprocessor.feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        # Reshape for LSTM (assuming single sequence)
        features_scaled = features_scaled.reshape(1, -1, features_scaled.shape[-1])
        
        return features_scaled
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with logging.
        
        Args:
            data: Input data
            
        Returns:
            Prediction result with metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess input
            X = self.preprocess_input(data)
            
            # Make prediction
            prediction = self.model.predict(X)[0][0]
            predicted_class = int(prediction > 0.5)
            
            inference_time = time.time() - start_time
            
            result = {
                'prediction_probability': float(prediction),
                'predicted_class': predicted_class,
                'inference_time_ms': inference_time * 1000,
                'timestamp': time.time()
            }
            
            # Log prediction
            logger.info(f"Prediction made: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

def train_pipeline(train_path: str, test_path: Optional[str] = None, 
                  sequence_length: int = 10, epochs: int = 100) -> Dict[str, Any]:
    """
    Complete training pipeline.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        sequence_length: LSTM sequence length
        epochs: Number of training epochs
        
    Returns:
        Training results and metrics
    """
    logger.info("Starting training pipeline")
    
    # Data preprocessing
    preprocessor = DataPreprocessor(sequence_length=sequence_length)
    train_df, test_df = preprocessor.load_data(train_path, test_path)
    processed_data = preprocessor.preprocess(train_df, test_df)
    
    # Train LSTM model
    lstm_model = LSTMClassifier(
        sequence_length=sequence_length,
        n_features=processed_data['n_features']
    )
    
    history = lstm_model.train(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_val'],
        processed_data['y_val'],
        epochs=epochs
    )
    
    # Evaluate LSTM model
    lstm_metrics = lstm_model.evaluate(
        processed_data['X_val'],
        processed_data['y_val']
    )
    
    # Train and evaluate baseline
    baseline = BaselineClassifier(method='altitude_threshold')
    baseline.fit(processed_data['X_train'], processed_data['y_train'])
    baseline_metrics = baseline.evaluate(
        processed_data['X_val'],
        processed_data['y_val']
    )
    
    # Save model and preprocessor
    lstm_model.save_model('missile_lstm_model.h5')
    joblib.dump(preprocessor.scaler, 'scaler.pkl')
    
    results = {
        'lstm_metrics': lstm_metrics,
        'baseline_metrics': baseline_metrics,
        'training_history': history,
        'model_path': 'missile_lstm_model.h5',
        'scaler_path': 'scaler.pkl'
    }
    
    logger.info("Training pipeline completed successfully")
    return results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Missile Flight Phase Classification')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True,
                       help='Mode: train or predict')
    parser.add_argument('--train_data', type=str, help='Path to training data')
    parser.add_argument('--test_data', type=str, help='Path to test data')
    parser.add_argument('--model_path', type=str, default='missile_lstm_model.h5',
                       help='Path to model file')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl',
                       help='Path to scaler file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=10, help='LSTM sequence length')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.train_data:
            raise ValueError("Training data path required for train mode")
        
        results = train_pipeline(
            train_path=args.train_data,
            test_path=args.test_data,
            sequence_length=args.sequence_length,
            epochs=args.epochs
        )
        
        # Save results
        with open('training_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if key == 'training_history':
                    serializable_results[key] = {k: [float(x) for x in v] for k, v in value.items()}
                elif isinstance(value, dict):
                    serializable_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                               for k, v in value.items()}
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2)
        
        print("Training completed. Results saved to training_results.json")
        print(f"LSTM Model Performance: {results['lstm_metrics']}")
        print(f"Baseline Performance: {results['baseline_metrics']}")
        
    elif args.mode == 'predict':
        # Example prediction service usage
        service = ModelInferenceService(args.model_path, args.scaler_path)
        
        # Example input data
        sample_data = {
            'timestamp': time.time(),
            'track_id': 1,
            'sensor_id': 1,
            'latitude': 45.0,
            'longitude': -120.0,
            'altitude': 50000.0,
            'radiometric_intensity': 0.8
        }
        
        result = service.predict(sample_data)
        print(f"Prediction result: {result}")

if __name__ == "__main__":
    main()