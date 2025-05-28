# Time Series Classification for Reentry Phase Prediction

## Overview

This project implements a time series classification model to predict the reentry phase of a missile based on simulated sensor data. The model utilizes an LSTM network with sensor ID embeddings to effectively classify whether a missile is in the reentry phase.

## Table of Contents

1.  Model Architecture and Rationale
2.  Data Preprocessing and Feature Engineering
3.  Scalability Considerations and Trade-offs
4.  Comparison to Baseline Performance
5.  Dependencies
6.  Usage
    - Training
    - Inference
7.  Using Docker And Kubernetes
8.  File Descriptions
9.  Running Unit Tests

## 1. Model Architecture and Rationale

### Model Architecture

The model is an `LSTMTimeStepClassifier` built using PyTorch. It consists of the following components:

- **Input Layer:** Takes two inputs:
  - `x_numeric`: Numeric features (latitude, longitude, altitude, radiometric intensity) of shape `(batch_size, seq_len, input_size)` where `input_size` is 4.
  - `x_sensor`: Sensor IDs, which are passed through an embedding layer, of shape `(batch_size, seq_len)`.
- **Sensor ID Embedding:** A PyTorch embedding layer that maps sensor IDs to a dense vector representation. This allows the model to learn sensor-specific patterns. The embedding dimension is configurable (`sensor_embed_dim`).
- **LSTM Layer:** A multi-layered LSTM network that processes the combined numeric features and sensor embeddings to capture temporal dependencies in the data. The number of layers (`num_layers`) and hidden state size (`hidden_size`) are configurable.
- **Dropout:** Dropout layers are applied for regularization to prevent overfitting.
- **Output Layer:** A fully connected linear layer that maps the LSTM output to the classification logits for each time step. The number of classes is 2 (reentry phase or not).

### Rationale

- **LSTMs for Time Series:** LSTMs are well-suited for time series data because they can learn long-range dependencies while avoiding issues of exploding or vanishing gradient.
- **Sensor ID Embeddings:** Different sensors might have unique characteristics or biases. Embedding sensor IDs allows the model to learn these characteristics and improve accuracy.
- **Time Step Classification:** The model predicts the reentry phase at each time step, providing a more granular understanding of when the transition occurs.

## 2. Data Preprocessing and Feature Engineering

The `data.py` module handles data loading and preprocessing. The following steps are applied:

- **Loading:** The data is loaded from a CSV file using `pandas`.
- **Sensor ID Encoding:** Sensor IDs (categorical) are encoded into numerical indices using `LabelEncoder`. This is necessary for the embedding layer. The encoder is saved to `sensor_encoder.pkl` for use during inference.
- **Outlier Clipping:** Numeric features (latitude, longitude, altitude, radiometric intensity) have their values clipped to a specified range (defined by `lower` and `upper` quantiles in the config) to reduce the impact of extreme outliers.
- **Smoothing:** A rolling mean smoothing is applied to the numeric features to reduce noise and highlight trends. A window size of 3 is used.
- **Scaling:** Numeric features are standardized using `StandardScaler` to have zero mean and unit variance. This helps with model convergence. The scaler is saved to `scaler.pkl` for inference.
- **Windowing:** The data is split into fixed-length sequences (windows) using a sliding window approach. Each window contains numeric features, sensor IDs, and corresponding labels. This is done by the `make_windows` and `create_windows` functions.
- **Dataset Creation:** A `TimeSeriesDataset` class is defined using PyTorch's `Dataset` class to efficiently load and batch the windowed data during training and evaluation.

## 3. Scalability Considerations and Trade-offs

- **Sequence Length (`seq_len`):** Longer sequences capture more context but increase computational cost and memory usage. Shorter sequences are computationally cheaper but might miss long-term dependencies.
- **Batch Size (`batch_size`):** Larger batch sizes can speed up training but require more memory. Smaller batch sizes can improve generalization but might slow down training.
- **LSTM Hidden Size (`hidden_size`) and Number of Layers (`num_layers`):** Larger hidden sizes and more layers increase model capacity but also increase the risk of overfitting and require more computation.
- **Sensor Embedding Dimension (`sensor_embed_dim`):** Larger embedding dimensions can capture more complex sensor-specific information but increase the number of model parameters.
- **Data Size:** The code uses batching and windowing to handle potentially large datasets. However, very large datasets might require distributed training or more sophisticated data loading strategies.
- **Inference Speed:** Inference speed is affected by the model size and sequence length. For real-time applications, model optimization techniques (e.g., quantization, pruning) might be needed.

## 4. Comparison to Baseline Performance

The `train.py` script includes a baseline model for comparison:

- **Baseline Model:** The baseline model predicts the reentry phase based on a simple rule: if the altitude decreases consecutively for a certain number of time steps (30 in this case), the model predicts that the object is in the reentry phase.

- **Rationale:** This baseline captures a key characteristic of reentry – a sustained decrease in altitude. It provides a simple benchmark to assess the effectiveness of the LSTM model.

- **Evaluation:** The training script evaluates both the LSTM model and the baseline on the validation set, reporting accuracy and F1-score, along with a classification report. This allows for a quantitative comparison of their performance.

- **Expected Outcome:** The LSTM model is expected to outperform the baseline by learning more complex patterns and considering other sensor inputs in addition to altitude.

## 5. Dependencies

- Python 3.7+
- PyTorch
- Pandas
- Scikit-learn
- Numpy

You can install the necessary packages using pip:

```bash
pip install torch pandas scikit-learn numpy
```

## 6. Usage

### Training

To train the model, run the `train.py` script:

```bash
python train.py
```

The script will:

1. Load and preprocess the training data.
2. Create training and validation datasets.
3. Define the LSTM model.
4. Train the model using the specified hyperparameters.
5. Evaluate the model on the validation set.
6. Save the best model weights to model.pth.

### Inference

To run inference on a test CSV file, use the inference.py script. Unlabled test data is located here: `/data/test.csv`

```bash
python inference.py --input <path_to_test_csv> --output <path_to_output_csv>
```

The script will:

1. Load the saved sensor encoder and scaler.
2. Load and preprocess the test data.
3. Load the trained model.
4. Generate predictions for each time step in the test data.
5. Save the predictions to a CSV file.

## 7. Using Docker And Kubernetes

### Build A Docker Image

Run this command in the root directory

```bash
docker build -t <image_name>:latest -f Dockerfile.txt .
```

### Load To Minikube

Start minikube and load the docker image into minikube using this command

```bash
minikube start
minikube image load <image_name>:latest
```

### Deply To Kubernetes

In the `deployment.yaml` file, change the volumes host path (`volumes:hostPath:path`) to the path of the root directory of the project. Then run:

```bash
kubectl apply -f deployment.yaml
```

Check your deployment:

```bash
kubectl get pods
kubectl logs <pod-name>
```

## 8. File Descriptions

- data.py: Contains functions for loading, preprocessing, and preparing the data for training and evaluation. Includes the TimeSeriesDataset class.
- model.py: Defines the LSTMTimeStepClassifier model architecture.
- train.py: Implements the training loop, evaluation logic, and baseline model comparison.
- inference.py: Handles loading the trained model and generating predictions on new data.
- config.py: This file contains configuration parameters such as file paths, hyperparameters (e.g., seq_len, batch_size, hidden_size, learning_rate), and data preprocessing parameters (e.g., lower, upper quantiles).
- app.log: Log file for training and inference information.
- sensor_encoder.pkl: Saved LabelEncoder object.
- scaler.pkl: Saved StandardScaler object.
- best_model.pt: Saved model weights.

## 9. Running Unit Tests

This project includes unit tests to ensure the correctness of the core components, including data processing, model architecture, and training logic. The tests are located in the `tests/` directory and cover:

- **Data Processing:** Validates window creation, preprocessing, and dataset loading.
- **Model:** Checks the LSTM model's forward pass and output shapes.
- **Training:** Verifies training loop components and loss calculations.

### Running Tests (Windows)

A batch script `run_tests.bat` is provided for convenience. It will run all tests and display a coverage report in the terminal.

To run the tests, simply execute:

```bash
run_tests.bat
```

This will:

- Run all unit tests using `pytest`
- Show a code coverage summary in the terminal
- Pause at the end so you can review the results

### Manual Test Execution

Alternatively, you can run the tests manually with:

```bash
python -m pytest --cov=. --cov-report=term
```

### Test Output

- All test results and coverage information will be displayed in your terminal.
- Ensure you have `pytest` and `pytest-cov` installed (add them to `requirements.txt` if needed):

```txt
pytest
pytest-cov
```
