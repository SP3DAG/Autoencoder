# Autoencoder

**Autoencoder** is a machine learning project using **TensorFlow/Keras** to decode hidden messages embedded within compressed images via **steganography**. This project leverages **autoencoders** to learn meaningful representations that help uncover concealed data, offering a step toward intelligent steganalysis.

---

## Project Overview

Steganography is the art of concealing information within digital media. When combined with image compression, detecting these hidden messages becomes increasingly difficult. This project implements a neural autoencoder architecture trained to:

- Learn efficient image encodings.
- Detect and decode messages hidden within compressed image data.
- Evaluate decoding accuracy and model performance.

---

## Project Structure

```
Autoencoder % Tree -L 2
├── README.md                   <- Project documentation
├── data/                       <- Scripts and directories for raw and processed data
│   ├── prepare_data.py         <- Prepares and processes datasets
│   ├── processed/              <- Preprocessed data
│   ├── raw/                    <- Raw image and stego data
│   └── test.py                 <- Data testing and inspection
├── evaluation/                
│   └── evaluate_model.py       <- Evaluation metrics and performance tests
├── logs/                       <- Training and evaluation logs
├── models/
│   ├── autoencoder.py          <- Core model architecture
│   ├── utils.py                <- Helper functions
├── requirements.txt            <- Dependencies list
├── saved_models/               <- Pre-trained or trained model weights
│   ├── autoencoder_v1.h5
│   └── autoencoder_regression.keras
├── training/
│   ├── config.py               <- Training configuration and hyperparameters
│   └── train.py                <- Model training script
└── venv/                       <- Virtual environment
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/SP3DAG/Autoencoder.git
cd Autoencoder
```

### 2. Set Up Environment

Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Model

The model is a deep autoencoder built with Keras, designed to reconstruct compressed images to the original version so that the steganography message is restored.

---

## Usage

### Train the model

```bash
python training/train.py
```

### Evaluate model performance

```bash
python evaluation/evaluate_model.py
```

---

## Logs & Results

Training and evaluation logs are saved in the `logs/` directory. You can use TensorBoard for visualization:

```bash
tensorboard --logdir=logs/
```

---

## Contributions

Feel free to open issues or submit pull requests for improvements, fixes, or feature ideas.