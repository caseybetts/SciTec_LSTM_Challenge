# Missile Flight Phase Classification System

## Overview

This is a comprehensive MLOps solution for classifying missile flight phases using LSTM neural networks. The system distinguishes between reentry and non-reentry phases based on sensor tracking data including position, altitude, and radiometric intensity measurements.

## Architecture

The solution consists of several key components:

- **Data Processing Pipeline**: Feature engineering and sequence creation for time-series data
- **LSTM Model**: Deep learning classifier for sequential missile tracking data
- **Baseline Models**: Simple classifiers for performance comparison
- **Production Inference Service**: Flask-based REST API with monitoring
- **Containerization**: Docker deployment with health checks
- **Kubernetes Orchestration**: Production-ready K8s manifests with auto-scaling
- **Monitoring**: Prometheus metrics and structured logging

## System Components

### 1. Core Machine Learning (`missile_classification_solution.py`)

#### DataPreprocessor Class
Handles data loading, preprocessing, and feature engineering for missile tracking data.

**Key Features:**
- Time-series feature engineering (velocity, acceleration, distance calculations)
- Sequence creation for LSTM input
- Data normalization and scaling
- Missing value handling

**Usage:**
```python
preprocessor = DataPreprocessor(sequence_length=10)
train_df, test_df = preprocessor.load_data('train.csv', 'test.csv')
processed_data = preprocessor.preprocess(train_df, test_df)
```

#### LSTMClassifier Class
Deep learning model for missile flight phase classification.

**Architecture:**
- Bidirectional LSTM layers with batch normalization
- Dropout for regularization
- Dense layers with sigmoid activation for binary classification
- Adam optimizer with learning rate scheduling

**Features:**
- Early stopping and model checkpointing
- Comprehensive evaluation metrics
- Model persistence

**Usage:**
```python
classifier = LSTMClassifier(sequence_length=10, n_features=11)
classifier.build_model()
classifier.train(X_train, y_train, X_val, y_val)
metrics = classifier.evaluate(X_test, y_test)
```

#### BaselineClassifier Class
Simple baseline models for performance comparison.

**Methods:**
- `altitude_threshold`: Threshold-based classification using altitude
- `random_forest`: Random Forest classifier on flattened sequences
- `dummy`: Most frequent class classifier

#### ModelInferenceService Class
Production-ready inference service with logging and monitoring.

**Features:**
- Real-time preprocessing and prediction
- Performance monitoring
- Error handling and logging
- Input validation

### 2. Web Service (`flask_app.py`)

Flask-based REST API for model inference with production features.

#### Endpoints

**Health Check**
```
GET /health
```
Returns service health status and readiness.

**Single Prediction**
```
POST /predict
Content-Type: application/json

{
  "latitude": 45.0,
  "longitude": -120.0,
  "altitude": 50000.0,
  "radiometric_intensity": 0.8,
  "timestamp": 1635724800,
  "track_id": 1,
  "sensor_id": 1
}
```

**Batch Prediction**
```
POST /batch_predict
Content-Type: application/json

{
  "samples": [
    {
      "latitude": 45.0,
      "longitude": -120.0,
      "altitude": 50000.0,
      "radiometric_intensity": 0.8
    }
  ]
}
```

**Prometheus Metrics**
```
GET /metrics
```
Returns Prometheus-formatted metrics for monitoring.

**Service Information**
```
GET /info
```
Returns service metadata and available endpoints.

#### Monitoring Features

- **Request Metrics**: Total requests, latency histograms
- **Prediction Metrics**: Prediction counts by class
- **Error Metrics**: Error counts by type
- **Structured Logging**: JSON-formatted logs with request tracking

### 3. Containerization (`dockerfile.txt`)

Production-ready Docker container with:

- **Base Image**: Python 3.9 slim for minimal attack surface
- **Security**: Non-root user execution
- **Health Checks**: Built-in container health monitoring
- **Optimization**: Multi-stage build and dependency caching

**Build and Run:**
```bash
docker build -t missile-classifier:latest .
docker run -p 8000:8000 missile-classifier:latest
```

### 4. Kubernetes Deployment (`kubernetes_manifests.txt`)

Complete Kubernetes deployment with production features:

#### Components

**Namespace**
- Isolated environment for the application

**ConfigMap**
- Environment configuration
- Model and scaler paths
- Service configuration

**Deployment**
- 3 replica pods for high availability
- Resource requests and limits
- Liveness and readiness probes
- Persistent volume for model storage

**Services**
- ClusterIP service for internal communication
- NodePort service for external access

**Horizontal Pod Autoscaler (HPA)**
- CPU and memory-based auto-scaling
- 2-10 replica range
- 70% CPU and 80% memory thresholds

**Ingress**
- External traffic routing
- Host-based routing configuration

**PersistentVolumeClaim**
- Shared storage for model files
- ReadOnlyMany access mode

**ServiceMonitor**
- Prometheus scraping configuration
- Metrics collection setup

#### Deployment Commands

```bash
# Apply all manifests
kubectl apply -f kubernetes_manifests.txt

# Check deployment status
kubectl get pods -n missile-classifier
kubectl get services -n missile-classifier

# Access service
kubectl port-forward svc/missile-classifier-service 8080:80 -n missile-classifier
```

### 5. Dependencies (`requirements_txt.txt`)

**Core ML Libraries:**
- TensorFlow 2.13.0 for deep learning
- NumPy, Pandas for data processing
- Scikit-learn for preprocessing and metrics
- Joblib for model persistence

**Web Framework:**
- Flask 2.3.2 for REST API
- Gunicorn for production WSGI server

**Monitoring:**
- Prometheus-client for metrics
- Structlog for structured logging

**Development:**
- Pytest for testing
- Black, Flake8 for code quality
- MyPy for type checking

### 6. Testing (`unit_tests.py`)

Comprehensive test suite covering:

#### Unit Tests
- **DataPreprocessor**: Data loading, feature engineering, sequence creation
- **LSTMClassifier**: Model building, training, prediction
- **BaselineClassifier**: All baseline methods
- **ModelInferenceService**: Component loading, prediction service

#### Integration Tests
- End-to-end training pipeline
- Service initialization and prediction flow
- Model persistence and loading

#### Test Execution

```bash
# Run all tests
python -m pytest unit_tests.py -v

# Run with coverage
python unit_tests.py --coverage

# Generate HTML coverage report
coverage html
```

## Data Format

### Input Data Schema

The system expects CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | int/float | Unix timestamp or sequential time |
| track_id | int | Unique identifier for each missile track |
| sensor_id | int | Sensor system identifier |
| latitude | float | Latitude coordinate (degrees) |
| longitude | float | Longitude coordinate (degrees) |
| altitude | float | Altitude (feet or meters) |
| radiometric_intensity | float | Sensor intensity measurement (0-1) |
| reentry_phase | int | Target label (0=non-reentry, 1=reentry) |

### Feature Engineering

The system automatically creates additional features:

- **Velocity Components**: First derivatives of position
- **Speed**: Magnitude of velocity vector
- **Acceleration**: Rate of change of speed
- **Distance from Origin**: Euclidean distance from (0,0,0)
- **Radiometric Rate**: Rate of change of intensity

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | /app/models/missile_lstm_model.h5 | Path to trained model |
| SCALER_PATH | /app/models/scaler.pkl | Path to fitted scaler |
| HOST | 0.0.0.0 | Service bind address |
| PORT | 8000 | Service port |
| LOG_LEVEL | INFO | Logging level |
| DEBUG | false | Debug mode flag |

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| sequence_length | 10 | LSTM input sequence length |
| lstm_units | 64 | LSTM layer units |
| epochs | 100 | Training epochs |
| batch_size | 32 | Training batch size |
| learning_rate | 0.001 | Adam optimizer learning rate |

## Usage Examples

### Training a New Model

```bash
# Train with default parameters
python missile_classification_solution.py \
  --mode train \
  --train_data data/train.csv \
  --test_data data/test.csv

# Train with custom parameters
python missile_classification_solution.py \
  --mode train \
  --train_data data/train.csv \
  --epochs 50 \
  --sequence_length 15
```

### Running Inference Service

```bash
# Start service locally
python flask_app.py

# Using Docker
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  missile-classifier:latest

# Using Kubernetes
kubectl apply -f kubernetes_manifests.txt
```

### Making Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 45.0,
    "longitude": -120.0,
    "altitude": 50000.0,
    "radiometric_intensity": 0.8
  }'

# Batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "latitude": 45.0,
        "longitude": -120.0,
        "altitude": 50000.0,
        "radiometric_intensity": 0.8
      }
    ]
  }'
```

## Performance Monitoring

### Metrics Available

**Request Metrics:**
- `missile_classifier_requests_total`: Total requests by method and endpoint
- `missile_classifier_request_duration_seconds`: Request latency histogram

**Prediction Metrics:**
- `missile_classifier_predictions_total`: Total predictions by class
- `missile_classifier_errors_total`: Total errors by type

### Monitoring Setup

The system integrates with Prometheus for metrics collection and can be monitored using:

- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **ServiceMonitor**: Kubernetes-native metrics scraping

### Log Analysis

Structured JSON logs include:
- Request/response details
- Prediction results and confidence
- Error tracking and debugging info
- Performance metrics

## Security Considerations

### Container Security
- Non-root user execution
- Minimal base image
- No unnecessary packages
- Health check endpoints

### Kubernetes Security
- Namespace isolation
- Resource limits and requests  
- ReadOnlyMany volume access
- Network policies (can be added)

### API Security
- Input validation and sanitization
- Error handling without information leakage
- Request rate limiting (can be added)
- Authentication/authorization (can be added)

## Troubleshooting

### Common Issues

**Model Loading Errors:**
- Check model and scaler file paths
- Verify file permissions in container
- Ensure persistent volume is mounted correctly

**Memory Issues:**
- Adjust resource limits in Kubernetes
- Optimize batch sizes for prediction
- Monitor memory usage metrics

**Performance Issues:**
- Check CPU/memory utilization
- Verify HPA scaling configuration
- Monitor request latency metrics

**Prediction Quality:**
- Validate input data format and ranges
- Check feature engineering consistency
- Compare with baseline model performance

### Debugging Commands

```bash
# Check pod logs
kubectl logs -f deployment/missile-classifier-deployment -n missile-classifier

# Check resource usage
kubectl top pods -n missile-classifier

# Check HPA status
kubectl get hpa -n missile-classifier

# Port forward for direct access
kubectl port-forward svc/missile-classifier-service 8080:80 -n missile-classifier
```

## Development Workflow

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd missile-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest unit_tests.py -v

# Code formatting
black .
flake8 .
mypy .
```

### Model Development Cycle

1. **Data Analysis**: Explore and understand the missile tracking data
2. **Feature Engineering**: Develop and test new features
3. **Model Training**: Experiment with architectures and hyperparameters
4. **Evaluation**: Compare against baselines and previous models
5. **Testing**: Run comprehensive test suite
6. **Deployment**: Update production service

### CI/CD Integration

The codebase is structured for easy CI/CD integration:

- **Testing**: Comprehensive unit and integration tests
- **Code Quality**: Linting and type checking
- **Security**: Container scanning and dependency checks
- **Deployment**: Kubernetes manifests for multiple environments

## Scaling and Production Considerations

### Horizontal Scaling
- HPA configuration for automatic scaling
- Load balancing across multiple replicas
- Stateless service design

### Performance Optimization
- Model quantization for faster inference
- Batch prediction endpoints for throughput
- Caching for repeated predictions

### Data Management
- Model versioning and rollback capabilities
- A/B testing framework for model comparison
- Feature store integration for consistent preprocessing

### Observability
- Distributed tracing for request flows
- Custom business metrics
- Alerting on model performance degradation

## Future Enhancements

### Model Improvements
- Ensemble methods for better accuracy
- Online learning for model updates
- Anomaly detection for outlier tracking data

### System Enhancements
- GraphQL API for flexible queries
- Real-time streaming predictions
- Multi-model serving architecture

### Operations
- GitOps deployment workflows
- Blue-green deployment strategies
- Canary releases for model updates

## Support and Maintenance

### Model Retraining
- Schedule regular retraining with new data
- Monitor for data drift and model decay
- Implement automated retraining triggers

### System Updates
- Regular security updates for base images
- Dependency updates and vulnerability scanning
- Performance optimization and profiling

### Documentation
- Keep API documentation updated
- Document model performance baselines
- Maintain operational runbooks

This documentation provides a comprehensive guide to understanding, deploying, and maintaining the missile flight phase classification system. The modular design allows for easy customization and extension based on specific requirements and deployment environments.