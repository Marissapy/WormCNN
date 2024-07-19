# WormCNN for C. elegans Age Classification

This repository hosts the WormCNN, a convolutional neural network designed for the age classification of C. elegans worms, alongside a preprocessing script to prepare the images for training and testing.

## Components

- `worm_preprocessing.py`: A script that preprocesses worm images by resampling skeleton coordinates, generating dense skeleton points, and straightening images.
- `worm_cnn.py`: The WormCNN model implementation in PyTorch, including data loading, model training, and evaluation processes.

## Installation

To set up this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Marissapy/WormCNN.git
cd WormCNN
pip install -r requirements.txt
