import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Preprocessing Function
def preprocess():
    """ Preprocesses the MNIST dataset. """
    mat = loadmat('mnist_all.mat')  # Load MNIST dataset

    n_feature = mat.get("train1").shape[1]
    n_sample = sum(mat.get(f"train{i}").shape[0] for i in range(10))
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct Validation Data and Labels
    validation_data = np.zeros((10 * n_validation, n_feature))
    validation_label = np.zeros((10 * n_validation, 1))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get(f"train{i}")[:n_validation, :]
        validation_label[i * n_validation:(i + 1) * n_validation, 0] = i

    # Construct Training Data and Labels
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp_index = 0
    for i in range(10):
        samples = mat.get(f"train{i}")
        n_samples = samples.shape[0] - n_validation
        train_data[temp_index:temp_index + n_samples, :] = samples[n_validation:, :]
        train_label[temp_index:temp_index + n_samples, 0] = i
        temp_index += n_samples

    # Construct Test Data and Labels
    n_test = sum(mat.get(f"test{i}").shape[0] for i in range(10))
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp_index = 0
    for i in range(10):
        samples = mat.get(f"test{i}")
        n_samples = samples.shape[0]
        test_data[temp_index:temp_index + n_samples, :] = samples
        test_label[temp_index:temp_index + n_samples, 0] = i

    # Remove Features with Near-Zero Variance
    std_devs = np.std(train_data, axis=0)
    useful_features = std_devs > 0.001
    train_data = train_data[:, useful_features]
    validation_data = validation_data[:, useful_features]
    test_data = test_data[:, useful_features]

    # Scale Features to [0, 1]
    train_data = train_data / 255.0
    validation_data = validation_data / 255.0
    test_data = test_data / 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


# Helper Functions
def sigmoid(z):
    """Computes the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """ Computes the softmax of input array z. """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Logistic Regression Objective Function
def blrObjFunction(weights, X, yk):
    """ Computes the error and gradient for Logistic Regression. """
    N = X.shape[0]
    X = np.hstack((np.ones((N, 1)), X))  # Add bias term
    weights = weights.reshape(-1, 1)  # Reshape weights to column vector
    theta = sigmoid(np.dot(X, weights))  # Predicted probabilities
    error = -np.mean(yk * np.log(theta) + (1 - yk) * np.log(1 - theta))  # Cross-entropy error
    error_grad = np.dot(X.T, (theta - yk)) / N  # Gradient
    return error, error_grad.flatten()


# Logistic Regression Prediction Function
def blrPredict(W, X):
    """ Predicts labels using logistic regression. """
    N = X.shape[0]
    X = np.hstack((np.ones((N, 1)), X))  # Add bias term
    probs = sigmoid(np.dot(X, W))  # Class probabilities
    label = np.argmax(probs, axis=1).reshape(-1, 1)  # Predicted class
    return label


# Multi-class Logistic Regression Objective Function
def mlrObjFunction(params, X, Y):
    """
    Computes the error and gradient for Multi-class Logistic Regression.
    """
    N = X.shape[0]
    K = Y.shape[1]
    X = np.hstack((np.ones((N, 1)), X))  # Add bias term
    params = params.reshape(X.shape[1], K)  # Reshape weights to (D + 1) x K
    logits = np.dot(X, params)  # Compute logits (N x K)
    probs = softmax(logits)  # Compute softmax probabilities (N x K)
    error = -np.sum(Y * np.log(probs)) / N  # Cross-entropy error
    error_grad = np.dot(X.T, (probs - Y)) / N  # Gradient
    return error, error_grad.flatten()


# Multi-class Logistic Regression Prediction
def mlrPredict(W, X):
    """
    Predicts labels using Multi-class Logistic Regression.
    """
    N = X.shape[0]
    X = np.hstack((np.ones((N, 1)), X))  # Add bias term
    logits = np.dot(X, W)  # Compute logits
    label = np.argmax(logits, axis=1).reshape(-1, 1)  # Predicted class
    return label


# Error Computation Function
def compute_total_and_class_error(W, X, Y, true_labels):
    """
    Computes total and per-class cross-entropy errors.
    """
    N = X.shape[0]
    X = np.hstack((np.ones((N, 1)), X))  # Add bias term
    logits = np.dot(X, W)  # Compute logits (N x K)
    probs = softmax(logits)  # Compute softmax probabilities (N x K)

    # Total error
    total_error = -np.sum(Y * np.log(probs)) / N

    # Per-class error
    class_errors = {}
    for k in range(Y.shape[1]):  # Loop through each class
        class_mask = (true_labels == k).flatten()  # Select samples for class k
        class_probs = probs[class_mask, :]  # Probabilities for class k
        class_labels = Y[class_mask, :]  # True labels for class k
        if class_probs.shape[0] > 0:  # Avoid division by zero
            class_error = -np.sum(class_labels * np.log(class_probs)) / class_mask.sum()
            class_errors[k] = class_error
        else:
            class_errors[k] = 0.0  # No samples for this class

    return total_error, class_errors


# Main Script
if __name__ == "__main__":
    # Preprocess Data
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # Number of classes and features
    n_class = 10
    n_train, n_feature = train_data.shape

    # One-hot encode labels for training and testing data
    Y = np.eye(n_class)[train_label.astype(int).flatten()]
    Y_test = np.eye(n_class)[test_label.astype(int).flatten()]

    # One-vs-All Logistic Regression
    W_ovr = np.zeros((n_feature + 1, n_class))  # Weight matrix for One-vs-All
    opts = {'maxiter': 100}
    for k in range(n_class):
        yk = (train_label == k).astype(int)  # Binary labels for class k
        initialWeights = np.zeros(n_feature + 1)  # Initialize weights
        result = minimize(blrObjFunction, initialWeights, jac=True, args=(train_data, yk), method='L-BFGS-B', options=opts)
        W_ovr[:, k] = result.x  # Store optimized weights

    # Multi-class Logistic Regression
    initialWeights = np.zeros((n_feature + 1) * n_class)  # Initialize weights
    result = minimize(mlrObjFunction, initialWeights, jac=True, args=(train_data, Y), method='L-BFGS-B', options=opts)
    W_mlr = result.x.reshape(n_feature + 1, n_class)  # Reshape optimized weights

    # Compute Errors for One-vs-All
    train_total_error_ovr, train_class_errors_ovr = compute_total_and_class_error(W_ovr, train_data, Y, train_label)
    test_total_error_ovr, test_class_errors_ovr = compute_total_and_class_error(W_ovr, test_data, Y_test, test_label)

    print("\nOne-vs-All Logistic Regression:")
    print(f"Total Training Error: {train_total_error_ovr:.4f}")
    print(f"Total Testing Error: {test_total_error_ovr:.4f}")
    print("Per-Class Training Errors:", train_class_errors_ovr)
    print("Per-Class Testing Errors:", test_class_errors_ovr)

    # Compute Errors for Multi-class
    train_total_error_mlr, train_class_errors_mlr = compute_total_and_class_error(W_mlr, train_data, Y, train_label)
    test_total_error_mlr, test_class_errors_mlr = compute_total_and_class_error(W_mlr, test_data, Y_test, test_label)

    print("\nMulti-class Logistic Regression:")
    print(f"Total Training Error: {train_total_error_mlr:.4f}")
    print(f"Total Testing Error: {test_total_error_mlr:.4f}")
    print("Per-Class Training Errors:", train_class_errors_mlr)
    print("Per-Class Testing Errors:", test_class_errors_mlr)

import matplotlib.pyplot as plt

# Data for visualization
strategies = ["One-vs-All", "Multi-class"]
total_training_errors = [train_total_error_ovr, train_total_error_mlr]
total_testing_errors = [test_total_error_ovr, test_total_error_mlr]

# Per-class errors
classes = list(range(10))
one_vs_all_training_errors = [v for v in train_class_errors_ovr.values()]
one_vs_all_testing_errors = [v for v in test_class_errors_ovr.values()]
multi_class_training_errors = [v for v in train_class_errors_mlr.values()]
multi_class_testing_errors = [v for v in test_class_errors_mlr.values()]

# Visualization 1: Total Training and Testing Errors
plt.figure(figsize=(10, 6))
plt.bar(strategies, total_training_errors, label='Training Error', alpha=0.7)
plt.bar(strategies, total_testing_errors, label='Testing Error', alpha=0.7)
plt.title("Total Training and Testing Errors")
plt.ylabel("Error")
plt.legend()
plt.show()

# Visualization 2: Per-Class Errors (Training)
plt.figure(figsize=(12, 6))
x = range(len(classes))
width = 0.2
plt.bar([p - width for p in x], one_vs_all_training_errors, width=width, label='One-vs-All (Training)')
plt.bar(x, multi_class_training_errors, width=width, label='Multi-class (Training)')
plt.xticks(ticks=x, labels=[f"Class {c}" for c in classes])
plt.title("Per-Class Training Errors")
plt.xlabel("Classes")
plt.ylabel("Error")
plt.legend()
plt.show()

# Visualization 3: Per-Class Errors (Testing)
plt.figure(figsize=(12, 6))
plt.bar([p - width for p in x], one_vs_all_testing_errors, width=width, label='One-vs-All (Testing)')
plt.bar(x, multi_class_testing_errors, width=width, label='Multi-class (Testing)')
plt.xticks(ticks=x, labels=[f"Class {c}" for c in classes])
plt.title("Per-Class Testing Errors")
plt.xlabel("Classes")
plt.ylabel("Error")
plt.legend()
plt.show()

