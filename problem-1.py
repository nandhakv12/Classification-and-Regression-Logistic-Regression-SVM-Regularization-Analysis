import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

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
        temp_index += n_samples

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


# Sigmoid Function
def sigmoid(z):
    """Computes the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


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


# Main Script for Problem 1
if __name__ == "__main__":
    # Preprocess Data
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # Number of classes and features
    n_class = 10
    n_train, n_feature = train_data.shape

    # One-vs-All Logistic Regression
    W = np.zeros((n_feature + 1, n_class))  # Weight matrix (D+1 x K)
    opts = {'maxiter': 100}

    # Train logistic regression for each class
    for k in range(n_class):
        yk = (train_label == k).astype(int)  # Binary labels for class k
        initialWeights = np.zeros(n_feature + 1)  # Initialize weights
        result = minimize(blrObjFunction, initialWeights, jac=True, args=(train_data, yk), method='L-BFGS-B', options=opts)
        W[:, k] = result.x  # Store optimized weights

    # Predict labels for training and testing data
    train_predicted = blrPredict(W, train_data)
    test_predicted = blrPredict(W, test_data)

    # Calculate accuracies
    train_accuracy = 100 * np.mean(train_predicted == train_label)
    test_accuracy = 100 * np.mean(test_predicted == test_label)

    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Testing Accuracy: {test_accuracy:.2f}%')
