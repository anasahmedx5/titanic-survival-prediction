# Titanic Survival Prediction – Neural Network from Scratch

## Project Overview

This project implements a neural network **from scratch (no high-level ML libraries like TensorFlow or PyTorch)** to predict survival outcomes on the Titanic dataset. It features a custom implementation of the **forward pass**, **backpropagation**, and the **Adam optimization algorithm**. The goal is to classify passengers as **survived (1)** or **not survived (0)** based on features like age, class, gender, and more.

## Design Overview

### Data Preprocessing
- **Features Used**:
  - `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
  - `Sex`: Encoded as 0 (female), 1 (male)
  - `Age`: Missing values filled with median age
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Fare`: Ticket fare (normalized)
- **Label**: `Survived` (0 or 1)

- **Preprocessing Steps**:
  - Handled missing values in `Age` and `Fare`
  - Encoded categorical features (like `Sex`)
  - Normalized features using Min-Max scaling
  - Split dataset into training and testing sets (80/20 split)

### Neural Network Architecture
- **Input Layer**: Size equals the number of features (e.g., 6)
- **Hidden Layer 1**: 20 neurons with ReLU activation
- **Hidden Layer 2**: 10 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)

### Optimizer
- **Adam Optimizer (Implemented from Scratch)**:
  - Combines momentum and RMSProp
  - Maintains first and second moment estimates
  - Applies bias correction
  - Parameters: `β1=0.9`, `β2=0.999`, `ε=1e-8`

### Training
- **Epochs**: 5000
- **Learning Rate**: 0.001
- **Batch Size**: Full batch gradient descent
- **Weight Initialization**: Random normal distribution

## Key Components

### Libraries Used
- `numpy`: For numerical operations
- `pandas`: For data loading and preprocessing
- `sklearn`: For splitting data and evaluation metrics
- `matplotlib`: For visualizations

### Process
1. Load and preprocess Titanic dataset
2. Encode and scale features
3. Initialize weights and biases for all layers
4. Implement forward pass using ReLU and sigmoid activations
5. Compute loss using binary cross-entropy
6. Perform backpropagation to compute gradients
7. Update weights and biases using Adam optimizer
8. Evaluate the model using accuracy, precision, recall, F1-score
9. Plot loss over epochs

## Algorithms Used
- **Neural Network (Manual Implementation)**:
  - 3-layer architecture (2 hidden layers + output)
  - Custom forward and backward propagation logic
- **Activation Functions**:
  - ReLU for hidden layers
  - Sigmoid for output layer
- **Adam Optimizer**:
  - Adaptive learning rate optimizer for stable convergence
- **Binary Cross-Entropy Loss**:
  - Appropriate for binary classification problems

## Visualization
- **Loss Curve**: Line plot showing training loss decreasing over 5000 epochs
- **Confusion Matrix**: Visual comparison of predicted vs. actual survival outcomes
- **Bar Chart**: Feature importance (based on gradient magnitudes)

## Conclusion
This project demonstrates how a neural network can be built and trained from scratch to perform binary classification tasks like Titanic survival prediction. By manually implementing the forward pass, backpropagation, and Adam optimization, we gain a deeper understanding of how modern deep learning systems work under the hood. The model achieves competitive accuracy and sets a solid foundation for expanding to deeper networks or more complex datasets in the future.
