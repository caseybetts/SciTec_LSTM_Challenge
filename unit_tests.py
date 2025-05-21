"""
Comprehensive unit tests for missile classification system.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import pytest
from src.missile_classifier import (
    DataPreprocessor, 
    LSTMClassifier, 
    BaselineClassifier,
    ModelInferenceService,
    train_pipeline
)

class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor(sequence_length=5)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'track_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'sensor_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'latitude': [45.0, 45.1, 45.2, 45.3, 45.4, 46.0, 46.1, 46.2, 46.3, 46.4],
            'longitude': [-120.0, -120.1, -120.2, -120.3, -120.4, -121.0, -121.1, -121.2, -121.3, -121.4],
            'altitude': [50000, 48000, 46000, 44000, 42000, 55000, 53000, 51000, 49000, 47000],
            'radiometric_intensity': [0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.8, 0.7, 0.6, 0.5],
            'reentry_phase': [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        })
    
    def test_load_data(self):
        """Test data loading functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            train_df, test_df = self.preprocessor.load_data(f.name)
            
            self.assertIsInstance(train_df, pd.DataFrame)
            self.assertEqual(len(train_df), 10)
            self.assertIsNone(test_df)
            
            os.unlink(f.name)
    
    def test_engineer_features(self):
        """Test feature engineering."""
        engineered_df = self.preprocessor.engineer_features(self.sample_data)
        
        # Check that new features are created
        expected_features = [
            'velocity_lat', 'velocity_lon', 'velocity_alt', 
            'speed', 'acceleration', 'distance_from_origin', 'radiometric_rate'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, engineered_df.columns)
        
        # Check that engineered features have correct length
        self.assertEqual(len(engineered_df), len(self.sample_data))
    
    def test_create_sequences(self):
        """Test sequence creation for LSTM."""
        engineered_df = self.preprocessor.engineer_features(self.sample_data)
        X_sequences, y_sequences = self.preprocessor.create_sequences(engineered_df)
        
        self.assertIsInstance(X_sequences, np.ndarray)
        self.assertIsInstance(y_sequences, np.ndarray)
        self.assertEqual(X_sequences.shape[1], self.preprocessor.sequence_length)
        self.assertEqual(X_sequences.shape[2], len(self.preprocessor.feature_columns))
        self.assertEqual(len(X_sequences), len(y_sequences))
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        result = self.preprocessor.preprocess(self.sample_data)
        
        required_keys = ['X_train', 'X_val', 'y_train', 'y_val', 'n_features', 'sequence_length']
        for key in required_keys:
            self.assertIn(key, result)
        
        self.assertIsInstance(result['X_train'], np.ndarray)
        self.assertIsInstance(result['y_train'], np.ndarray)
        self.assertEqual(result['sequence_length'], self.preprocessor.sequence_length)

class TestLSTMClassifier(unittest.TestCase):
    """Test cases for LSTMClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sequence_length = 5
        self.n_features = 11
        self.classifier = LSTMClassifier(
            sequence_length=self.sequence_length,
            n_features=self.n_features
        )
        
        # Create dummy data
        self.X_train = np.random.random((100, self.sequence_length, self.n_features))
        self.y_train = np.random.randint(0, 2, 100)
        self.X_val = np.random.random((20, self.sequence_length, self.n_features))
        self.y_val = np.random.randint(0, 2, 20)
    
    def test_build_model(self):
        """Test model building."""
        model = self.classifier.build_model()
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, self.sequence_length, self.n_features))
        self.assertEqual(model.output_shape, (None, 1))
    
    @patch('tensorflow.keras.models.Sequential.fit')
    def test_train(self, mock_fit):
        """Test model training."""
        # Mock the fit method to avoid actual training
        mock_history = Mock()
        mock_history.history = {'loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
        mock_fit.return_value = mock_history
        
        self.classifier.build_model()
        history = self.classifier.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=2, batch_size=32
        )
        
        self.assertIn('loss', history)
        self.assertIn('val_loss', history)
    
    def test_predict_without_model(self):
        """Test prediction without trained model."""
        with self.assertRaises(ValueError):
            self.classifier.predict(self.X_val)
    
    @patch('tensorflow.keras.models.Sequential.predict')
    def test_predict_with_model(self, mock_predict):
        """Test prediction with model."""
        mock_predict.return_value = np.random.random((20, 1))
        
        self.classifier.build_model()
        predictions = self.classifier.predict(self.X_val)
        
        self.assertIsInstance(predictions, np.ndarray)
        mock_predict.assert_called_once()

class TestBaselineClassifier(unittest.TestCase):
    """Test cases for BaselineClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X = np.random.random((100, 5, 11))
        self.y = np.random.randint(0, 2, 100)
    
    def test_altitude_threshold_baseline(self):
        """Test altitude threshold baseline."""
        baseline = BaselineClassifier(method='altitude_threshold')
        baseline.fit(self.X, self.y)
        
        predictions = baseline.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsNotNone(baseline.threshold)
    
    def test_random_forest_baseline(self):
        """Test random forest baseline."""
        baseline = BaselineClassifier(method='random_forest')
        baseline.fit(self.X, self.y)
        
        predictions = baseline.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsNotNone(baseline.model)
    
    def test_dummy_baseline(self):
        """Test dummy classifier baseline."""
        baseline = BaselineClassifier(method='dummy')
        baseline.fit(self.X, self.y)
        
        predictions = baseline.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertIsNotNone(baseline.model)
    
    def test_evaluate(self):
        """Test baseline evaluation."""
        baseline = BaselineClassifier(method='altitude_threshold')
        baseline.fit(self.X, self.y)
        
        metrics = baseline.evaluate(self.X, self.y)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)

class TestModelInferenceService(unittest.TestCase):
    """Test cases for ModelInferenceService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = 'dummy_model.h5'
        self.scaler_path = 'dummy_scaler.pkl'
    
    @patch('tensorflow.keras.models.load_model')
    @patch('joblib.load')
    def test_load_components(self, mock_joblib_load, mock_load_model):
        """Test component loading."""
        mock_model = Mock()
        mock_scaler = Mock()
        mock_load_model.return_value = mock_model
        mock_joblib_load.return_value = mock_scaler
        
        service = ModelInferenceService(self.model_path, self.scaler_path)
        
        self.assertEqual(service.model, mock_model)
        self.assertEqual(service.scaler, mock_scaler)
    
    @patch('tensorflow.keras.models.load_model')
    @patch('joblib.load')
    def test_predict(self, mock_joblib_load, mock_load_model):
        """Test prediction service."""
        # Mock components
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.7]])
        
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((1, 11))
        
        mock_load_model.return_value = mock_model
        mock_joblib_load.return_value = mock_scaler
        
        service = ModelInferenceService(self.model_path, self.scaler_path)
        
        sample_data = {
            'latitude': 45.0,
            'longitude': -120.0,
            'altitude': 50000.0,
            'radiometric_intensity': 0.8
        }
        
        with patch.object(service, 'preprocess_input') as mock_preprocess:
            mock_preprocess.return_value = np.random.random((1, 5, 11))
            
            result = service.predict(sample_data)
            
            self.assertIn('prediction_probability', result)
            self.assertIn('predicted_class', result)
            self.assertIn('inference_time_ms', result)
            self.assertIn('timestamp', result)

class TestTrainingPipeline(unittest.TestCase):
    """Test cases for training pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary CSV files
        self.sample_data = pd.DataFrame({
            'timestamp': range(20),
            'track_id': [1] * 10 + [2] * 10,
            'sensor_id': [1] * 20,
            'latitude': np.random.uniform(40, 50, 20),
            'longitude': np.random.uniform(-130, -110, 20),
            'altitude': np.random.uniform(40000, 60000, 20),
            'radiometric_intensity': np.random.uniform(0.1, 1.0, 20),
            'reentry_phase': np.random.randint(0, 2, 20)
        })
        
        self.train_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.train_file.name, index=False)
        self.train_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.train_file.name)
        
        # Clean up any generated files
        for file in ['missile_lstm_model.h5', 'scaler.pkl', 'training_results.json']:
            if os.path.exists(file):
                os.unlink(file)
    
    @patch('src.missile_classifier.LSTMClassifier.train')
    @patch('src.missile_classifier.LSTMClassifier.evaluate')
    @patch('src.missile_classifier.LSTMClassifier.save_model')
    def test_train_pipeline(self, mock_save, mock_evaluate, mock_train):
        """Test complete training pipeline."""
        # Mock training results
        mock_train.return_value = {'loss': [0.5, 0.4], 'val_loss': [0.6, 0.5]}
        mock_evaluate.return_value = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'auc_roc': 0.90
        }
        
        with patch('joblib.dump'):
            results = train_pipeline(
                train_path=self.train_file.name,
                sequence_length=5,
                epochs=2
            )
        
        self.assertIn('lstm_metrics', results)
        self.assertIn('baseline_metrics', results)
        self.assertIn('training_history', results)
        self.assertIn('model_path', results)
        self.assertIn('scaler_path', results)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create a larger, more realistic dataset
        np.random.seed(42)
        n_samples = 200
        n_tracks = 10
        
        data = []
        for track_id in range(1, n_tracks + 1):
            track_length = n_samples // n_tracks
            
            # Simulate missile trajectory
            base_lat = np.random.uniform(40, 50)
            base_lon = np.random.uniform(-130, -110)
            base_alt = np.random.uniform(50000, 70000)
            
            for i in range(track_length):
                # Simulate descent for reentry phase
                reentry = 1 if i > track_length * 0.6 else 0
                alt_factor = 0.95 if reentry else 1.0
                
                data.append({
                    'timestamp': i,
                    'track_id': track_id,
                    'sensor_id': 1,
                    'latitude': base_lat + np.random.normal(0, 0.1),
                    'longitude': base_lon + np.random.normal(0, 0.1),
                    'altitude': base_alt * (alt_factor ** i) + np.random.normal(0, 1000),
                    'radiometric_intensity': np.random.uniform(0.1, 1.0),
                    'reentry_phase': reentry
                })
        
        self.test_data = pd.DataFrame(data)
        
        # Save to temporary file
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.test_file.name, index=False)
        self.test_file.close()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        os.unlink(self.test_file.name)
        
        # Clean up generated files
        for file in ['missile_lstm_model.h5', 'scaler.pkl', 'best_model.h5']:
            if os.path.exists(file):
                os.unlink(file)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Run training pipeline with minimal epochs for speed
        results = train_pipeline(
            train_path=self.test_file.name,
            sequence_length=5,
            epochs=2
        )
        
        # Verify results structure
        self.assertIn('lstm_metrics', results)
        self.assertIn('baseline_metrics', results)
        
        # Verify metrics are reasonable
        lstm_metrics = results['lstm_metrics']
        baseline_metrics = results['baseline_metrics']
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            self.assertIn(metric, lstm_metrics)
            self.assertIn(metric, baseline_metrics)
            self.assertGreaterEqual(lstm_metrics[metric], 0.0)
            self.assertLessEqual(lstm_metrics[metric], 1.0)
        
        # Verify model files are created
        self.assertTrue(os.path.exists('missile_lstm_model.h5'))
        self.assertTrue(os.path.exists('scaler.pkl'))

if __name__ == '__main__':
    # Run tests with coverage
    import sys
    
    if '--coverage' in sys.argv:
        import coverage
        
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        unittest.main(argv=sys.argv[:-1], exit=False)
        
        cov.stop()
        cov.save()
        
        print("\n" + "="*50)
        print("COVERAGE REPORT")
        print("="*50)
        cov.report()
        
        # Generate HTML report
        cov.html_report(directory='htmlcov')
        print(f"\nDetailed HTML coverage report generated in 'htmlcov' directory")
    else:
        unittest.main()