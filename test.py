import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.utils import resample
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def preprocess():
    """
    Input:
    Although this function doesn't have any input, you are required to load
    the MNIST data set from file 'mnist_all.mat'.

    Output:
    train_data: matrix of training set. Each row of train_data contains
      feature vector of a image
    train_label: vector of label corresponding to each image in the training
      set
    validation_data: matrix of validation set. Each row of validation_data
      contains feature vector of a image
    validation_label: vector of label corresponding to each image in the
      validation set
    test_data: matrix of testing set. Each row of test_data contains
      feature vector of a image
    test_label: vector of label corresponding to each image in the testing
      set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample += mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp += size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test += mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp += size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([i for i in range(n_feature) if sigma[i] > 0.001])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    Logistic Regression Objective Function
    Computes error and gradient for binary logistic regression.
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]

    # Reshape initialWeights into a column vector
    weights = initialWeights.reshape((n_feature + 1, 1))

    # Add bias term
    train_data = np.hstack((np.ones((n_data, 1)), train_data))

    theta = sigmoid(np.dot(train_data, weights))

    # Compute cross-entropy error
    error = -np.mean(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))

    # Compute gradient
    error_grad = np.dot(train_data.T, (theta - labeli)) / n_data

    return error, error_grad.flatten()


def blrPredict(W, data):
    """
    Predict Labels using Logistic Regression Weights
    """
    n_data = data.shape[0]

    # Add bias term
    data = np.hstack((np.ones((n_data, 1)), data))

    probs = sigmoid(np.dot(data, W))
    label = np.argmax(probs, axis=1).reshape(-1, 1)
    return label


def mlrObjFunction(params, *args):
    # [Same as before]
    train_data, Y = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = Y.shape[1]

    # Reshape params into a weight matrix of size (D + 1) x K
    W = params.reshape((n_feature + 1, n_class))

    # Add bias term
    train_data = np.hstack((np.ones((n_data, 1)), train_data))

    # Compute logits
    logits = np.dot(train_data, W)

    # Stabilize logits for numerical accuracy
    logits -= np.max(logits, axis=1, keepdims=True)

    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute cross-entropy error
    error = -np.sum(Y * np.log(probabilities)) / n_data

    # Compute gradient
    error_grad = np.dot(train_data.T, (probabilities - Y)) / n_data

    return error, error_grad.flatten()


def mlrPredict(W, data):
    # [Same as before]
    n_data = data.shape[0]

    # Add bias term
    data = np.hstack((np.ones((n_data, 1)), data))

    # Compute logits
    logits = np.dot(data, W)

    # Compute softmax probabilities
    logits -= np.max(logits, axis=1, keepdims=True)  # Numerical stability
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Predict labels using argmax of probabilities
    label = np.argmax(probabilities, axis=1).reshape(-1, 1)
    return label


if __name__ == "__main__":
    """
    Script for Logistic Regression
    """
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # Number of classes
    n_class = 10

    # Number of training samples
    n_train = train_data.shape[0]

    # Number of features
    n_feature = train_data.shape[1]

    # Create label matrix for one-vs-all logistic regression
    Y = np.zeros((n_train, n_class))
    for i in range(n_class):
        Y[:, i] = (train_label == i).astype(int).ravel()

    # Logistic Regression with Gradient Descent
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros((n_feature + 1, 1))  # Initialize as 2D array

    opts = {'maxiter': 100}
    for i in range(n_class):
        labeli = Y[:, i].reshape(n_train, 1)
        args = (train_data, labeli)
        # Flatten initialWeights before passing to minimize
        nn_params = minimize(blrObjFunction, initialWeights.flatten(), jac=True, args=args, method='CG', options=opts)
        # Reshape the optimized parameters back to (n_feature + 1, 1) and store
        W[:, i] = nn_params.x.reshape((n_feature + 1,))

    # Find the accuracy on Training Dataset
    predicted_label = blrPredict(W, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    predicted_label = blrPredict(W, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    predicted_label = blrPredict(W, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

    """
    Script for Support Vector Machine
    """

    print('\n\n--------------SVM-------------------\n\n')

    # Randomly sample 10,000 data points with a fixed random_state
    train_data_sampled, train_label_sampled = resample(train_data, train_label, n_samples=10000, random_state=42)

    # SVM with Linear Kernel
    print("Training SVM with Linear Kernel...")
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(train_data_sampled, train_label_sampled.ravel())
    print("Training Accuracy:", 100 * svm_linear.score(train_data_sampled, train_label_sampled.ravel()), "%")
    print("Validation Accuracy:", 100 * svm_linear.score(validation_data, validation_label.ravel()), "%")
    print("Testing Accuracy:", 100 * svm_linear.score(test_data, test_label.ravel()), "%")

    # SVM with RBF Kernel (gamma=1)
    print("\nTraining SVM with RBF Kernel (gamma=1)...")
    svm_rbf_gamma1 = SVC(kernel='rbf', gamma=1.0, random_state=42)
    svm_rbf_gamma1.fit(train_data_sampled, train_label_sampled.ravel())
    print("Training Accuracy:", 100 * svm_rbf_gamma1.score(train_data_sampled, train_label_sampled.ravel()), "%")
    print("Validation Accuracy:", 100 * svm_rbf_gamma1.score(validation_data, validation_label.ravel()), "%")
    print("Testing Accuracy:", 100 * svm_rbf_gamma1.score(test_data, test_label.ravel()), "%")

    # SVM with RBF Kernel (default gamma)
    print("\nTraining SVM with RBF Kernel (default gamma)...")
    svm_rbf_default = SVC(kernel='rbf', random_state=42)
    svm_rbf_default.fit(train_data_sampled, train_label_sampled.ravel())
    print("Training Accuracy:", 100 * svm_rbf_default.score(train_data_sampled, train_label_sampled.ravel()), "%")
    print("Validation Accuracy:", 100 * svm_rbf_default.score(validation_data, validation_label.ravel()), "%")
    print("Testing Accuracy:", 100 * svm_rbf_default.score(test_data, test_label.ravel()), "%")

    # SVM with RBF Kernel (varying C)
    print("\nTraining SVM with RBF Kernel (default gamma) and varying C...")
    C_values = [1, 10, 20, 30, 40, 50, 70, 100]
    training_accuracies = []
    validation_accuracies = []
    testing_accuracies = []

    for C in C_values:
        print(f"Training with C = {C}...")
        svm_rbf = SVC(kernel='rbf', C=C, random_state=42)
        svm_rbf.fit(train_data_sampled, train_label_sampled.ravel())
        training_accuracy = 100 * svm_rbf.score(train_data_sampled, train_label_sampled.ravel())
        validation_accuracy = 100 * svm_rbf.score(validation_data, validation_label.ravel())
        testing_accuracy = 100 * svm_rbf.score(test_data, test_label.ravel())
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)
        testing_accuracies.append(testing_accuracy)
        print(f"Training Accuracy: {training_accuracy}%")
        print(f"Validation Accuracy: {validation_accuracy}%")
        print(f"Testing Accuracy: {testing_accuracy}%")

    # Plot Accuracy vs. C
    plt.figure()
    plt.plot(C_values, validation_accuracies, marker='o')
    plt.title("Validation Accuracy vs. C (RBF Kernel, default gamma)")
    plt.xlabel("C Value")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid()
    plt.show()

    """
    Script for Extra Credit Part
    """

    # FOR EXTRA CREDIT ONLY
    print("\n\n--------------Multi-class Logistic Regression-------------------\n\n")

    initialWeights_b = np.zeros((n_feature + 1, n_class))  # Initialize as 2D array

    opts_b = {'maxiter': 100}

    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b.flatten(), jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))

    # Find the accuracy on Training Dataset
    predicted_label_b = mlrPredict(W_b, train_data)
    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

    # Find the accuracy on Validation Dataset
    predicted_label_b = mlrPredict(W_b, validation_data)
    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

    # Find the accuracy on Testing Dataset
    predicted_label_b = mlrPredict(W_b, test_data)
    print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
