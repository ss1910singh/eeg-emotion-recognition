# EEG-Based Emotion Recognition Using Deep Learning

## Overview

This project implements an advanced emotion recognition system using Electroencephalography (EEG) data and deep learning techniques. By leveraging Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Deep Neural Network (DNN) models, we aim to classify emotions with high accuracy, targeting 98.44% or higher.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Features

- EEG signal preprocessing and feature extraction
- Implementation of LSTM, GRU, and DNN models for emotion classification
- High accuracy emotion recognition (target: 98.44%+)
- Support for multiple EEG datasets (e.g., DEAP, SEED)
- Real-time emotion classification capabilities

## Installation

```bash
git clone https://github.com/yourusername/eeg-emotion-recognition.git
cd eeg-emotion-recognition
pip install -r requirements.txt
```

<!-- ## Usage

To train the model:

```bash
python train.py --data_path /path/to/eeg/data --model lstm
```

To evaluate the model:

```bash
python evaluate.py --model_path /path/to/saved/model --test_data /path/to/test/data
```

For real-time emotion classification:

```bash
python realtime_classify.py --model_path /path/to/saved/model
``` -->

## Data

This project supports various EEG datasets, including:

- DEAP (Database for Emotion Analysis using Physiological Signals)
- SEED (SJTU Emotion EEG Dataset)

Ensure your data is in the correct format and update the `config.py` file with the appropriate data paths.

## Model Architecture

Our emotion recognition system employs three main deep learning architectures:

1. LSTM (Long Short-Term Memory)
2. GRU (Gated Recurrent Unit)
3. DNN (Deep Neural Network)

These models are designed to capture temporal dependencies in EEG signals and extract relevant features for emotion classification.

## Future Work

- Implement multimodal emotion recognition by incorporating other physiological signals
- Explore transfer learning techniques for improved generalization
- Develop a user-friendly GUI for real-time emotion monitoring

## Contributing

We welcome contributions to improve the project. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
