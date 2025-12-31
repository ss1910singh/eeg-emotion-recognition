
# 🚀 EEG-Based Emotion Recognition Using Deep Learning

## Overview

This project implements an advanced emotion recognition system using Electroencephalography (EEG) data and deep learning techniques.  
By leveraging LSTM, GRU, DNN, and SNN models, the system classifies emotions from EEG signals, achieving high accuracy with LSTM as the best-performing model.

---

## Features

- **EEG Signal Preprocessing and Feature Extraction**: Extracts meaningful features from EEG signals using FFT, time-domain features, filtering, and artifact removal.  
- **High-Accuracy Emotion Recognition**: LSTM-based system achieves the highest accuracy among all tested models.  
- **Support for Multiple EEG Datasets**: DEAP and SEED datasets supported.  
- **Real-Time Inference**: System supports checkpointing, early stopping, and modular deployment.  
- **Visualization & Performance Analysis**: Provides confusion matrices and performance metrics for model evaluation.  

---

## Installation

```bash
git clone https://github.com/ss1910singh/eeg-emotion-recognition.git
cd eeg-emotion-recognition
pip install -r requirements.txt
````

---

## Project Structure

```
eeg-emotion-recognition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── models/
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

---

## Data

Supports multiple EEG datasets:

* **DEAP**: Database for Emotion Analysis using Physiological Signals.
* **SEED**: SJTU Emotion EEG Dataset.

Raw data in `data/raw/`, processed data in `data/processed/`.

---

## Preprocessing

* Bandpass filtering for noise removal.
* ICA-based artifact removal.
* Feature extraction: FFT, time-domain features.
* StandardScaler normalization.
* One-hot encoding for labels.
* Train-test-validation split.

Scripts located in `src/preprocessing/`.

---

## Model Architecture

* **LSTM**: 256 units, return sequences, Flatten + Dense with softmax.
* **GRU**: 256 units, return sequences, Flatten + Dense with softmax.
* **DNN**: Multiple Dense layers with ReLU, BatchNorm + Dropout, Dense with softmax.
* **SNN**: Spiking neuron dynamics, surrogate gradient descent, Dense with softmax.

Best performance achieved with **LSTM**.

---

## Training

```bash
python src/train.py --model lstm --data_path data/processed/DEAP --epochs 50
```

* Early stopping and checkpointing implemented.
* Learning rate scheduling used for optimal performance.

---

## Evaluation

```bash
python src/evaluate.py --model_path models/best_lstm_model.h5 --test_data data/processed/DEAP/test
```

---

## Results

* **Best Model:** LSTM
* **Accuracy:** 92.5%
* **Precision:** 0.90
* **Recall:** 0.89
* **F1 Score:** 0.89
* **ROC-AUC:** 0.95

> Confusion matrices and detailed visualizations are generated for further analysis.

---

## Repository

[GitHub Link](https://github.com/ss1910singh/eeg-emotion-recognition)

---

## Future Work

* Multimodal emotion recognition using additional physiological signals.
* Real-time GUI for live EEG emotion detection.
* Explore hybrid architectures like CNN-LSTM and STRNN for improved performance.
* Edge deployment for wearable EEG devices.
