# Neural Network from Scratch

A complete implementation of a feedforward neural network using only NumPy, demonstrating the fundamental concepts behind deep learning without relying on high-level frameworks.

## Overview

This project implements a neural network from the ground up to solve a real-world binary classification problem: detecting malignant breast cancer tumors. By building every component manually, from activation functions to backpropagation, this project reveals the mathematical foundations that frameworks like TensorFlow and PyTorch abstract away.

## Motivation

While modern deep learning frameworks make it easy to build and train models, they often hide the underlying mechanics. This project was created to understand what actually happens during training: how gradients flow backward through layers, how weights update during optimization, and why design choices like activation functions and initialization strategies matter.

## What's Implemented

### Core Components

**Activation Functions**
- Sigmoid: Outputs probabilities between 0 and 1
- ReLU (Rectified Linear Unit): Prevents vanishing gradients
- Tanh: Zero-centered outputs for better gradient flow

Each includes both the forward function and its derivative for backpropagation.

**Loss Functions**
- Binary Cross-Entropy: Measures prediction error for classification
- Mean Squared Error: Alternative loss function for comparison

**Weight Initialization**
- Random initialization: Small random values
- Xavier initialization: Optimized for sigmoid/tanh activations
- He initialization: Optimized for ReLU activations

**Neural Network Class**
A complete implementation featuring:
- Forward propagation through multiple layers
- Backpropagation using the chain rule
- Gradient descent optimization
- Training loop with validation monitoring
- Prediction with configurable thresholds

## The Dataset

The project uses the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains measurements from digitized images of cell nuclei from breast masses. The task is to classify tumors as malignant or benign based on 30 numerical features.

- 569 total samples
- 30 features per sample
- Binary classification: Malignant (1) or Benign (0)

## Results

The network achieves strong performance on this medical diagnosis task:

```
Architecture: 30 input features → 16 hidden → 8 hidden → 1 output
Training Accuracy: 98.49%
Validation Accuracy: 100%
Test Accuracy: 96.51%
Total Parameters: 641
```

After 2000 training epochs, the model successfully learned to distinguish between malignant and benign tumors, demonstrating that even a relatively simple architecture can achieve high accuracy when properly trained.

## Key Insights

**Why Activation Functions Matter**

Without non-linear activation functions, stacking multiple layers offers no advantage over a single layer. The network would collapse into one linear transformation, unable to learn complex patterns. Activation functions introduce the non-linearity that allows neural networks to approximate any continuous function.

**The Vanishing Gradient Problem**

Sigmoid and tanh activations suffer from vanishing gradients at extreme input values, where their derivatives approach zero. This makes learning slow or impossible in deep networks. ReLU addresses this by maintaining a constant gradient of 1 for positive inputs.

**The Role of Validation Data**

The validation set serves as an unbiased monitor during training, helping detect overfitting before evaluating on the test set. Without it, hyperparameter tuning on the test set would lead to overly optimistic performance estimates.

## Project Structure

```
Neural_Network_From_Scratch.ipynb    Main implementation notebook
requirements.txt                      Python dependencies
README.md                            This file
```

## Getting Started

**Installation**

```bash
pip install -r requirements.txt
```

**Usage**

```python
# Create a neural network
nn = NeuralNetwork(
    layer_dims=[30, 16, 8, 1],
    activation='relu',
    initialization='he'
)

# Train on your data
nn.train(
    X_train, y_train,
    X_val, y_val,
    epochs=2000,
    learning_rate=0.01
)

# Make predictions
predictions, probabilities = nn.predict(X_test)
```

## Technical Details

**Forward Propagation**

For each layer, the network computes:
1. Linear transformation: Z = W·X + b
2. Non-linear activation: A = activation(Z)

**Backpropagation**

Using the chain rule, gradients are computed backward through the network:
1. Output layer: dZ = A - y (for BCE loss with sigmoid)
2. Hidden layers: dZ = dA · activation'(Z)
3. Weight gradients: dW = (1/m) · dZ · A_prev^T
4. Bias gradients: db = (1/m) · sum(dZ)

**Gradient Descent Update**

Weights are updated to minimize loss:
- W = W - learning_rate · dW
- b = b - learning_rate · db

## Dependencies

- NumPy: Core numerical operations and linear algebra
- Matplotlib: Visualization of training curves and activations
- Scikit-learn: Dataset loading, preprocessing, and evaluation metrics
- Seaborn: Enhanced visualizations
- Pandas: Data manipulation
- Jupyter: Interactive notebook environment

## What I Learned

Building this network from scratch transformed my understanding of deep learning. Concepts that seemed abstract became concrete: backpropagation is just the chain rule applied systematically, gradient descent is simple iterative optimization, and the "learning" in machine learning is mathematical minimization.

The process also highlighted why frameworks exist. Implementing backpropagation manually for even a simple network requires careful bookkeeping and debugging. Production systems need the efficiency, automatic differentiation, and GPU acceleration that frameworks provide.

## Future Improvements

- Implement mini-batch gradient descent for faster training
- Add advanced optimizers (Adam, RMSprop)
- Include regularization techniques (L2, dropout)
- Extend to multi-class classification
- Visualize decision boundaries and learned features

## License

This project is open source and available for educational purposes.

## Acknowledgments

Dataset provided by the UCI Machine Learning Repository and available through scikit-learn. This implementation was built for educational purposes to understand the fundamentals of neural networks.
