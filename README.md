# Eye State Detection using CNN

A deep learning project that detects whether eyes are open or closed using Convolutional Neural Networks (CNN) and real-time webcam detection.

## Overview

This project trains a CNN model to classify eye states (Open/Closed) and implements real-time detection using a webcam feed. The model achieves 99.66% accuracy on the test dataset.

## Features

- Custom CNN architecture for binary eye state classification
- Data augmentation for improved model generalization
- Real-time eye detection and classification using webcam
- Comprehensive model evaluation with confusion matrix and classification reports
- Training visualization (accuracy and loss curves)

## Requirements

```
numpy
opencv-python (cv2)
scikit-learn
tensorflow
keras
matplotlib
seaborn
```

## Installation

```bash
pip install numpy opencv-python scikit-learn tensorflow keras matplotlib seaborn
```

## Dataset Structure

The project expects the following directory structure:

```
dataset/
└── train/
    ├── Closed/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── Open/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Model Architecture

The CNN consists of:

- **Convolutional Layers**: 4 Conv2D layers (16, 32, 64, 128 filters) with ReLU activation
- **Pooling Layers**: MaxPooling2D after each convolutional layer
- **Fully Connected Layers**: 2 Dense layers (64, 128 neurons) with ReLU activation
- **Output Layer**: Dense layer with softmax activation for binary classification

**Input Shape**: (64, 64, 1) - Grayscale images resized to 64x64 pixels

## Training

The model is trained with:

- **Data Augmentation**: Rotation, width/height shift, zoom, horizontal flip
- **Optimizer**: Adam
- **Loss Function**: Categorical crossentropy
- **Epochs**: 20
- **Batch Size**: 32
- **Train/Test Split**: 80/20

### Training Results

- **Final Training Accuracy**: ~98.97%
- **Final Validation Accuracy**: 99.66%
- **Test Accuracy**: 99.66%

## Usage

### 1. Train the Model

Run the training script to train the model on your dataset:

```python
python train_model.py
```

The model will be trained and evaluation metrics will be displayed.

### 2. Real-time Detection

To use the trained model for real-time eye state detection via webcam:

```python
python real_time_detection.py
```

**Note**: Uncomment the model loading line if you've saved your model:
```python
model = load_model("cnn_eye_model.keras")
```

### Controls

- Press **'q'** to quit the webcam window

## How It Works

1. **Preprocessing**: Images are converted to grayscale and resized to 64x64 pixels
2. **Normalization**: Pixel values are normalized to the range [0, 1]
3. **Data Augmentation**: Training data is augmented to prevent overfitting
4. **Training**: CNN model learns to distinguish between open and closed eyes
5. **Real-time Detection**: 
   - Haar Cascade detects eye regions in webcam frames
   - Detected regions are preprocessed and fed to the model
   - Model predicts eye state (Open/Closed)
   - Results are displayed with bounding boxes and labels

## Model Performance

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Closed | 1.00      | 0.99   | 1.00     | 148     |
| Open   | 0.99      | 1.00   | 1.00     | 143     |

**Overall Accuracy**: 99.66%

## Visualization

The project generates the following visualizations:

- **Confusion Matrix**: Shows the classification performance
- **Training Curves**: Accuracy and loss over epochs for both training and validation sets

## Potential Applications

- Driver drowsiness detection systems
- Attention monitoring in educational settings
- Accessibility tools for assistive technology
- Gaming and virtual reality interactions
- Security and surveillance systems

## Troubleshooting

### Protobuf Warnings
The protobuf version warnings can be safely ignored or suppressed. They indicate a version mismatch but don't affect functionality.

### Webcam Not Working
- Ensure your webcam is connected and accessible
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for external webcams
- Check if other applications are using the webcam

### Poor Detection Performance
- Ensure adequate lighting conditions
- Position your face directly in front of the camera
- Adjust the `scaleFactor` and `minNeighbors` parameters in `detectMultiScale()`

## Future Improvements

- Add drowsiness alert system based on consecutive closed eye detections
- Implement eye aspect ratio (EAR) for more robust detection
- Train on larger, more diverse datasets
- Add support for detecting both eyes simultaneously
- Deploy as a web application or mobile app

## License

This project is open source and available for educational purposes.

## Acknowledgments

- OpenCV for Haar Cascade classifiers
- TensorFlow/Keras for deep learning framework
- The open-source community for various tools and libraries
