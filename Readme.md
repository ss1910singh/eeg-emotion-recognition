
# 🚀EEG-Based Emotion Recognition Using Deep Learning

## Overview

This project implements an advanced emotion recognition system using Electroencephalography (EEG) data and deep learning techniques. By leveraging Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Deep Neural Network (DNN) models, we can achieve high accuracy in classifying emotions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Features

- EEG signal preprocessing and feature extraction
- Implementation of LSTM, GRU, and DNN models for emotion classification
- High accuracy emotion recognition
- Support for multiple EEG datasets (e.g., DEAP, SEED)
- Comprehensive data visualization and model performance analysis
- Real-time emotion classification capabilities

## Installation

```
git clone https://github.com/ss1910singh/eeg-emotion-recognition.git
cd eeg-emotion-recognition
pip install -r requirements.txt
```

## Project Structure

```
eeg-emotion-recognition/
│
├── data/
│   ├── raw/
│   │   ├── DEAP/
│   │   └── SEED/
│   └── processed/
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── signal_processing.py
│   │   └── feature_extraction.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   └── dnn.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── visualization.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_comparison.ipynb
│
├── config.py
├── requirements.txt
└── README.md
```

## Data

This project supports various EEG datasets, including:

- DEAP (Database for Emotion Analysis using Physiological Signals)
- SEED (SJTU Emotion EEG Dataset)

Raw data is stored in the `data/raw/` directory. After preprocessing, the processed data is saved in `data/processed/` for use by the models.

## Preprocessing

The preprocessing pipeline includes the following steps:

1. Signal filtering (bandpass filter)
2. Artifact removal (ICA-based approach)
3. Feature extraction (FFT, time-domain features)
4. Data normalization (StandardScaler)
5. Label encoding and one-hot encoding
6. Train-test split and validation set creation

Preprocessing scripts are located in `src/preprocessing/`.

## Model Architecture

Our emotion recognition system employs three main deep learning architectures:

1. LSTM (Long Short-Term Memory)
   - 256 LSTM units with return sequences
   - Flatten layer
   - Dense layer with softmax activation

2. GRU (Gated Recurrent Unit)
   - 256 GRU units with return sequences
   - Flatten layer
   - Dense layer with softmax activation

3. DNN (Deep Neural Network)
   - Multiple Dense layers with ReLU activation
   - Batch Normalization and Dropout for regularization
   - Final Dense layer with softmax activation

Model implementations can be found in `src/models/`.

## Training

To train the models:

```
python src/train.py --model lstm --data_path data/processed/DEAP --epochs 50
```

Training utilizes early stopping, model checkpointing, and learning rate scheduling for optimal performance.

## Evaluation

To evaluate the models:

```
python src/evaluate.py --model_path models/best_lstm_model.h5 --test_data data/processed/DEAP/test
```

## Results

Current model performance:
- LSTM: Results pending
- GRU: Results pending
- DNN: Results pending

Detailed performance metrics, confusion matrices, and visualizations are generated for each model.

## Future Work

- Implement multimodal emotion recognition by incorporating other physiological signals (e.g., ECG, GSR)
- Explore transfer learning techniques for improved generalization across datasets
- Develop a user-friendly GUI for real-time emotion monitoring
- Investigate attention mechanisms to improve model interpretability
- Implement ensemble methods to combine predictions from multiple models
- Optimize models for edge deployment (e.g., TensorFlow Lite conversion)
- Explore hybrid architectures like CNN-LSTM for improved performance
- Implement Graph CNN with LSTM for leveraging graph domain features
- Investigate Spatial-Temporal Recurrent Neural Network (STRNN) for integrated feature learning

## Contributing

We welcome contributions to improve the project. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code adheres to the project's coding standards and includes appropriate tests and documentation.