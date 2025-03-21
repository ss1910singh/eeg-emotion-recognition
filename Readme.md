# 🚀 EEG-Based Emotion Recognition Using Deep Learning

## Overview

This project implements an advanced emotion recognition system using Electroencephalography (EEG) data and deep learning techniques. By leveraging Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Deep Neural Network (DNN), and Spiking Neural Network (SNN) models, we aim to achieve high accuracy in classifying emotions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
<!--- [Future Work](#future-work)-->
<!--- [Contributing](#contributing)-->

## Features

- **EEG Signal Preprocessing and Feature Extraction**: The project includes a comprehensive preprocessing pipeline to extract meaningful features from EEG signals.
- **Implementation of LSTM, GRU, DNN, and SNN Models**: These models are used for emotion classification, providing a robust framework for handling temporal and spatial dependencies in EEG data.
- **High Accuracy Emotion Recognition**: The system aims to achieve high accuracy in emotion classification by leveraging the strengths of deep learning models.
- **Support for Multiple EEG Datasets**: The project supports various EEG datasets, including DEAP and SEED, allowing for flexibility and generalizability across different datasets.
- **Comprehensive Data Visualization and Model Performance Analysis**: Detailed visualizations and performance metrics are provided to understand model behavior and optimize performance.
- **Real-time Emotion Classification Capabilities**: The system is designed to classify emotions in real-time, making it suitable for applications requiring immediate feedback.

## Installation

To set up the project, follow these steps:

```
git clone https://github.com/ss1910singh/eeg-emotion-recognition.git
cd eeg-emotion-recognition
pip install -r requirements.txt
```

## Project Structure

The project is organized as follows:

```
eeg-emotion-recognition/
│
├── data/
│   ├── raw/
│   │   ├── CSV
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── dnn.py
│   │   └── snn.py
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

- **DEAP (Database for Emotion Analysis using Physiological Signals)**: A multimodal dataset for emotion analysis.
- **SEED (SJTU Emotion EEG Dataset)**: A dataset focused on EEG signals for emotion recognition.

Raw data is stored in the `data/raw/` directory. After preprocessing, the processed data is saved in `data/processed/` for use by the models.

## Exploratory Data Analysis (EDA)

### Completed Tasks

- **Feature Selection**: Used mutual information to select top features.
- **Dimensionality Reduction**: Planned for later stages.
- **Correlation Analysis**: Performed on a subset of features.
- **Visualization**: Used t-SNE for data visualization.

### Tasks to Complete

- **Detailed Feature Importance Analysis**: To be done after model training.

## Preprocessing

The preprocessing pipeline includes the following steps:

1. **Signal Filtering**: Bandpass filtering to remove noise.
2. **Artifact Removal**: ICA-based approach.
3. **Feature Extraction**: FFT, time-domain features.
4. **Data Normalization**: StandardScaler.
5. **Label Encoding**: One-hot encoding.
6. **Train-Test Split**: Creation of validation set.

Preprocessing scripts are located in `src/preprocessing/`.

### Completed Tasks

- **Missing Value Handling**: Imputed with mean.
- **Label Encoding**: Completed.
- **Data Splitting**: Done.

## Model Architecture

Our emotion recognition system employs four main deep learning architectures:

1. **LSTM (Long Short-Term Memory)**
   - 256 LSTM units with return sequences
   - Flatten layer
   - Dense layer with softmax activation

2. **GRU (Gated Recurrent Unit)**
   - 256 GRU units with return sequences
   - Flatten layer
   - Dense layer with softmax activation

3. **DNN (Deep Neural Network)**
   - Multiple Dense layers with ReLU activation
   - Batch Normalization and Dropout for regularization
   - Final Dense layer with softmax activation

4. **SNN (Spiking Neural Network)**
   - Leverages spiking neuron dynamics for temporal feature extraction.
   - Uses surrogate gradient descent for training.
   - Final Dense layer with softmax activation.

Model implementations can be found in `src/models/`.

## Training

To train the models:

```
python src/train.py --model lstm --data_path data/processed/DEAP --epochs 50
```

Training utilizes early stopping, model checkpointing, and learning rate scheduling for optimal performance.

### Completed Tasks

- **Checkpointing**: Implemented to track progress.

## Evaluation

To evaluate the models:

```
python src/evaluate.py --model_path models/best_lstm_model.h5 --test_data data/processed/DEAP/test
```

### Completed Tasks

- **Model Performance Metrics**: To be calculated after training.

## Results

Current model performance:
- **LSTM**: Results pending  
- **GRU**: Results pending  
- **DNN**: Results pending  
- **SNN**: Results pending  

Detailed performance metrics, confusion matrices, and visualizations are generated for each model.

<!--## Future Work-->

<!--Planned improvements include:-->
  
<!--1. Multimodal Emotion Recognition  -->
<!--2. Transfer Learning  -->
<!--3. Real-time GUI Development  -->
<!--4. Attention Mechanisms  -->
<!--5. Ensemble Methods  -->
<!--6. Edge Deployment  -->
<!--7. Hybrid Architectures like CNN-LSTM or Graph CNN-LSTM  -->
<!--8. Exploration of STRNN (Spatial Temporal Recurrent Neural Network).  -->