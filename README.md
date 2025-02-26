# ASL Recognition using PyTorch

This project is designed to recognize **American Sign Language (ASL)** gestures using a custom Convolutional Neural Network (CNN) implemented in **PyTorch**. The model is trained on a dataset of ASL images and can classify gestures into 35 categories (A-Z, space, del, and nothing).

Dataset :- https://drive.google.com/drive/folders/1bwRNtQ3ntn2IlIXYEOhNQm6CMW0J_ZOM?usp=sharing
---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Usage](#usage)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview
The goal of this project is to build a deep learning model that can recognize ASL gestures from images. The model is trained on a dataset of ASL images and uses a custom CNN architecture. The project includes:
- Data preprocessing and augmentation.
- A custom CNN model implemented in PyTorch.
- Training and evaluation scripts.
- Support for GPU acceleration.

---

## Features
- **Custom CNN Architecture**: A deep CNN model with multiple convolutional and fully connected layers.
- **Data Augmentation**: Image transformations like resizing, normalization, and random augmentations to improve model generalization.
- **GPU Support**: Utilizes CUDA for GPU acceleration (if available).
- **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving.
- **Progress Bar**: Displays training progress using `tqdm`.

---

## Installation
Follow these steps to set up the project on your local machine.

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (optional but recommended for faster training)
- CUDA Toolkit (if using GPU)
- cuDNN (if using GPU)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/asl-recognition-pytorch.git
   cd asl-recognition-pytorch
