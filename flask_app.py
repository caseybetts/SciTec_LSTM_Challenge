"""
Flask web service for missile classification inference.
Production-ready service with monitoring, logging, and health checks.
"""

import os
import time
import logging
from typing import Dict, Any
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog
from src.missile_classifier import ModelInferenceService

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('missile_classifier_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('missile_classifier_request_duration_seconds', 'Request latency')
PREDICTION_COUNT = Counter('missile_classifier_predictions_total', 'Total predictions', ['predicted_class'])
ERROR_COUNT = Counter('missile_classifier_errors_total', 'Total errors', ['error_type'])

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global inference service
inference_service = None

def initialize_service():
    """Initialize the model inference service."""
    global inference_service
    
    model_path = os.getenv('MODEL_PATH', 'models/missile_lstm_model.h5')
    scaler_path = os.getenv('SCALER_PATH', 'models/scaler.pkl')
    
    try:
        inference_service = ModelInferenceService(model_path, scaler_path)
        logger.info("Inference service initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize inference service", error=str(e))
        raise

@app.before_first_request
def startup():
    """Initialize service on first request."""
    initialize_service()

@app.before_request
def before_request():
    """Log request details."""
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Log response details and metrics."""
    if hasattr(request, 'start_time'):
        REQUEST_LATENCY.observe(time.time() - request.start_time)
    
    logger.info("Request processed",
                method=request.method,
                endpoint=request.endpoint,
                status_code=response.status_code,
                duration=time.time() - getattr(request, 'start_time', time.time()))
    
    return response

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        if inference_service is None:
            return jsonify({
                'status': 'unhealthy',
                'message': 'Inference service not initialized'
            }), 503
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'service': 'missile-classifier'
        })
    except Exception as e:
        ERROR_COUNT.labels(error_type='health_check').inc()
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'altitude', 'radiometric_intensity']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Add default values for optional fields
        data.setdefault('timestamp', time.time())
        data.setdefault('track_id', 1)
        data.setdefault('sensor_id', 1)
        
        # Make prediction
        result = inference_service.predict(data)
        
        # Update metrics
        PREDICTION_COUNT.labels(predicted_class=result['predicted_class']).inc()
        
        logger.info("Prediction made",
                    input_data=data,
                    prediction=result['predicted_class'],
                    probability=result['prediction_probability'],
                    inference_time_ms=result['inference_time_ms'])
        
        return jsonify(result)
        
    except ValueError as e:
        ERROR_COUNT.labels(error_type='validation').inc()
        logger.error("Validation error", error=str(e))
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        ERROR_COUNT.labels(error_type='prediction').inc()
        logger.error("Prediction error", error=str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple samples."""
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        if 'samples' not in data or not isinstance(data['samples'], list):
            return jsonify({'error': 'Request must contain "samples" list'}), 400
        
        results = []
        for i, sample in enumerate(data['samples']):
            try:
                # Add default values
                sample.setdefault('timestamp', time.time())
                sample.setdefault('track_id', i + 1)
                sample.setdefault('sensor_id', 1)
                
                result = inference_service.predict(sample)
                results.append(result)
                
                PREDICTION_COUNT.labels(predicted_class=result['predicted_class']).inc()
                
            except Exception as e:
                logger.error(f"Error processing sample {i}", error=str(e))
                results.append({'error': str(e)})
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        ERROR_COUNT.labels(error_type='batch_prediction').inc()
        logger.error("Batch prediction error", error=str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.route('/info', methods=['GET'])
def info():
    """Service information endpoint."""
    return jsonify({
        'service': 'missile-flight-phase-classifier',
        'version': '1.0.0',
        'model_type': 'LSTM',
        'description': 'Classifies missile flight phases into reentry and non-reentry',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Single prediction',
            '/batch_predict': 'Batch predictions',
            '/metrics': 'Prometheus metrics',
            '/info': 'Service information'
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    ERROR_COUNT.labels(error_type='internal').inc()
    logger.error("Internal server error", error=str(error))
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize service
    initialize_service()
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    logger.info("Starting missile classifier service",
                host=host,
                port=port,
                debug=debug)
    
    app.run(host=host, port=port, debug=debug)