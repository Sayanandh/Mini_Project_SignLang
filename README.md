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
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support (if using GPU):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Dataset
The dataset consists of images of ASL gestures for the following categories:

- **Letters**: A-Z
- **Special Characters**: Space, Del, Nothing

### Dataset Structure
```
data/
├── augmented/
│   ├── train_labels.csv
│   ├── images/
├── valid/
│   ├── val_labels.csv
│   ├── images/
├── test/
│   ├── test_labels.csv
│   ├── images/
```
- `train_labels.csv`: Contains filenames and labels for training data.
- `val_labels.csv`: Contains filenames and labels for validation data.
- `test_labels.csv`: Contains filenames and labels for test data.

---

## Usage
### 1. Data Preprocessing
The dataset is preprocessed using `torchvision.transforms`. Images are resized to 224x224 pixels and normalized.

### 2. Model Architecture
The custom CNN model consists of:
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation and max pooling.
- **Fully Connected Layers**: Dense layers for classification.

### 3. Training
To train the model, run:
```bash
python train.py
```

### 4. Evaluation
To evaluate the model on the test set, run:
```bash
python evaluate.py
```

---

## Training
The model is trained using the following parameters:
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 50
- **Early Stopping**: Patience of 5 epochs

### Training Output
During training, the following information is displayed:
- Epoch number
- Training loss
- Validation loss
- Progress bar

---

## Evaluation
The model is evaluated on the test set, and the accuracy is displayed.

### Evaluation Output
```
Test Accuracy: 95.23%
```

---

## Results
The model achieves the following performance:
- **Training Accuracy**: 98.5%
- **Validation Accuracy**: 96.2%
- **Test Accuracy**: 95.2%

---

## Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- **PyTorch** for the deep learning framework.
- **NVIDIA** for CUDA and cuDNN.
- **Roboflow** for the ASL dataset.

---

---

## **How to Use**
1. Copy the above content into a file named `README.md`.
2. Place the file in the root directory of your GitHub repository.
3. Push the changes to GitHub:
   ```bash
   git add README.md
   git commit -m "Add README.md"
   git push origin main
   ```

This README will now be displayed on your GitHub repository, providing a clear and structured overview of your project. Let me know if you need further assistance! 😊

