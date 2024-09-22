# Facial Expression Recognition Using a Custom CNN Model on FER2013 Dataset

### Author: Fareed Khan
### BSc Thesis - Abasyn University, Department of Computing

---

## Overview

This repository contains the implementation of a custom Convolutional Neural Network (CNN) designed to perform Facial Expression Recognition (FER) using the FER2013 dataset. The research focuses on creating a novel architecture optimized for the task, evaluating its performance against standard benchmarks, and highlighting the challenges of facial expression classification.

### Key Objectives

1. Design a CNN architecture specifically tailored for FER.
2. Train the model on the FER2013 dataset, ensuring effective feature extraction from facial images.
3. Evaluate the model's performance on various facial expressions and compare it with other CNN architectures.
4. Address challenges like overfitting and class imbalance, using techniques such as data augmentation and dropout.

## Dataset: FER2013

The FER2013 dataset is a widely used benchmark for facial expression recognition tasks. It consists of 35,887 grayscale images, each of size 48x48 pixels, labeled with seven different emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### Dataset Breakdown
- **Training set**: 28,709 images
- **Validation set**: 3,589 images
- **Test set**: 3,589 images

More details about the dataset can be found [here](https://www.kaggle.com/datasets/msambare/fer2013).

## Model Architecture

The custom CNN model developed for this research consists of multiple convolutional and pooling layers, followed by fully connected layers for classification. The architecture has been designed to strike a balance between computational efficiency and accuracy.

### Architecture Summary

1. **Input Layer**: 48x48 grayscale facial images.
2. **Convolutional Layers**: Multiple layers with increasing filter sizes (e.g., 32, 64, 128), kernel sizes of 3x3, and ReLU activations.
3. **Pooling Layers**: Max pooling layers (2x2) to reduce dimensionality and retain important features.
4. **Dropout**: Used after pooling to prevent overfitting.
5. **Fully Connected Layers**: Dense layers for feature interpretation, followed by a softmax output layer for classification.
6. **Output Layer**: 7 classes representing the facial expressions.

### Techniques Used
- **Batch Normalization**: To stabilize and speed up the training process.
- **Data Augmentation**: Random horizontal flips, rotations, and shifts to increase the robustness of the model.
- **Regularization**: Dropout layers and L2 regularization to reduce overfitting.


## Training and Evaluation

### Training Parameters
- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam Optimizer with learning rate scheduling
- **Batch Size**: 64
- **Epochs**: 100 (with early stopping)
- **Validation Split**: 20% of the training set used for validation

### Metrics
- **Accuracy**: Evaluated across the training, validation, and test sets.
- **Confusion Matrix**: To analyze the classification performance for each emotion class.
- **F1 Score**: To assess the balance between precision and recall, especially in handling class imbalances.

## Conclusion

This project demonstrates the effectiveness of a custom CNN model for Facial Expression Recognition using the FER2013 dataset. The model achieves competitive accuracy with improvements in handling class imbalances through data augmentation and regularization techniques. Future work could focus on deploying this model in real-time applications or exploring transfer learning approaches to further improve performance.

## Future Work

1. **Transfer Learning**: Experimenting with pre-trained models such as VGG16, ResNet50, or Inception.
2. **Improvement of Class Imbalance**: Further improving the model’s performance on underrepresented classes like 'Disgust' and 'Fear.'

---

## References

1. [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)