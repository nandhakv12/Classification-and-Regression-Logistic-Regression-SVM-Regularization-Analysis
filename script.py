import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def preprocess():
    """
    Preprocess the MNIST dataset.

    Output:
    - train_data: Training data matrix.
    - train_label: Training data labels.
    - validation_data: Validation data matrix.
    - validation_label: Validation data labels.
    - test_data: Test data matrix.
    - test_label: Test data labels.
    """
    mat = loadmat('mnist_all.mat')  # Load the MAT object as a dictionary

    # Initialize variables
    n_feature = mat.get("train1").shape[1]
    n_validation = 1000
    n_train = sum([mat.get("train" + str(i)).shape[0] for i in range(10)]) - 10 * n_validation

    # Construct validation data and labels
    validation_data = np.zeros((10 * n_validation, n_feature))
    validation_label = np.zeros((10 * n_validation, 1))
    for i in range(10):
        # Extract validation samples for each class
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[:n_validation, :]
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i

    # Construct training data and labels
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        # Extract training samples (excluding validation samples) for each class
        size_i = mat.get("train" + str(i)).shape[0]
        train_samples = mat.get("train" + str(i))[n_validation:size_i, :]
        train_data[temp:temp + size_i - n_validation, :] = train_samples
        train_label[temp:temp + size_i - n_validation, :] = i
        temp += size_i - n_validation

    # Construct test data and labels
    n_test = sum([mat.get("test" + str(i)).shape[0] for i in range(10)])
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        # Extract test samples for each class
        size_i = mat.get("test" + str(i)).shape[0]
        test_samples = mat.get("test" + str(i))
        test_data[temp:temp + size_i, :] = test_samples
        test_label[temp:temp + size_i, :] = i
        temp += size_i

    # Delete features with low variance (threshold = 0.001)
    sigma = np.std(train_data, axis=0)
    index = np.where(sigma > 0.001)[0]
    train_data = train_data[:, index]
    validation_data = validation_data[:, index]
    test_data = test_data[:, index]

    # Scale data to [0, 1]
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def sigmoid(z):
    """
    Compute the sigmoid function.
    """
    return 1.0 / (1.0 + np.exp(-z))

def blrObjFunction(initialWeights, *args):
    """
    Binary Logistic Regression Objective Function.

    Inputs:
    - initialWeights: Initial weights vector.
    - args: Tuple containing (train_data, labeli).

    Outputs:
    - error: The scalar value of the error function.
    - error_grad: The gradient vector of the error function.
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]

    # Reshape initialWeights into a column vector
    weights = initialWeights.reshape((n_feature + 1, 1))

    # Add bias term to the data
    train_data_bias = np.hstack((np.ones((n_data, 1)), train_data))

    # Compute the sigmoid of the linear combination
    theta = sigmoid(np.dot(train_data_bias, weights))

    # Numerical stability: Add epsilon to avoid log(0)
    epsilon = 1e-5
    theta = np.clip(theta, epsilon, 1 - epsilon)

    # Compute the error function (cross-entropy)
    error = -np.mean(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))

    # Compute the gradient of the error function
    error_grad = np.dot(train_data_bias.T, (theta - labeli)) / n_data
    error_grad = error_grad.flatten()

    return error, error_grad

def blrPredict(W, data):
    """
    Predict labels using the trained weights for Binary Logistic Regression.

    Inputs:
    - W: Weight matrix of size (n_feature + 1) x n_class.
    - data: Data matrix of size n_data x n_feature.

    Outputs:
    - label: Predicted labels vector.
    """
    n_data = data.shape[0]

    # Add bias term to the data
    data_bias = np.hstack((np.ones((n_data, 1)), data))

    # Compute probabilities using sigmoid function
    probs = sigmoid(np.dot(data_bias, W))

    # Predict the class with the highest probability
    label = np.argmax(probs, axis=1).reshape(-1, 1)

    return label

def mlrObjFunction(params, *args):
    """
    Multi-Class Logistic Regression Objective Function.

    Inputs:
    - params: Weight vector of size (n_feature + 1) x n_class.
    - args: Tuple containing (train_data, Y).

    Outputs:
    - error: The scalar value of the error function.
    - error_grad: The gradient vector of the error function.
    """
    train_data, Y = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    n_class = Y.shape[1]

    # Reshape params into a weight matrix
    W = params.reshape((n_feature + 1, n_class))

    # Add bias term to the data
    train_data_bias = np.hstack((np.ones((n_data, 1)), train_data))

    # Compute logits
    logits = np.dot(train_data_bias, W)

    # Numerical stability: Subtract max logit
    logits -= np.max(logits, axis=1, keepdims=True)

    # Compute softmax probabilities
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Numerical stability: Add epsilon to avoid log(0)
    epsilon = 1e-5
    probs = np.clip(probs, epsilon, 1 - epsilon)

    # Compute the error function (cross-entropy)
    error = -np.sum(Y * np.log(probs)) / n_data

    # Compute the gradient of the error function
    error_grad = np.dot(train_data_bias.T, (probs - Y)) / n_data
    error_grad = error_grad.flatten()

    return error, error_grad

def mlrPredict(W, data):
    """
    Predict labels using the trained weights for Multi-Class Logistic Regression.

    Inputs:
    - W: Weight matrix of size (n_feature + 1) x n_class.
    - data: Data matrix of size n_data x n_feature.

    Outputs:
    - label: Predicted labels vector.
    """
    n_data = data.shape[0]

    # Add bias term to the data
    data_bias = np.hstack((np.ones((n_data, 1)), data))

    # Compute logits
    logits = np.dot(data_bias, W)

    # Numerical stability: Subtract max logit
    logits -= np.max(logits, axis=1, keepdims=True)

    # Compute softmax probabilities
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Predict the class with the highest probability
    label = np.argmax(probs, axis=1).reshape(-1, 1)

    return label

if __name__ == "__main__":
    """
    Main script for training and evaluating models.
    """

    # Preprocess the data
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # Number of classes and features
    n_class = 10
    n_feature = train_data.shape[1]

    # Convert labels to one-hot encoding for multi-class logistic regression
    Y = np.zeros((train_data.shape[0], n_class))
    for i in range(n_class):
        Y[:, i] = (train_label.ravel() == i).astype(int)

    # ----------------- Logistic Regression (One-vs-All) -----------------
    print('\n\n-------------- Logistic Regression (One-vs-All) --------------\n')

    # Initialize weights and optimization settings
    W = np.zeros((n_feature + 1, n_class))
    initialWeights = np.zeros((n_feature + 1, 1))
    opts = {'maxiter': 100}

    # Train logistic regression classifiers for each class
    for i in range(n_class):
        print(f'Training class {i} vs all...')
        labeli = Y[:, i].reshape(-1, 1)
        args = (train_data, labeli)
        nn_params = minimize(blrObjFunction, initialWeights.flatten(), jac=True, args=args, method='CG', options=opts)
        W[:, i] = nn_params.x.reshape((n_feature + 1,))

    # Predict labels for training, validation, and testing datasets
    predicted_label_train = blrPredict(W, train_data)
    predicted_label_validation = blrPredict(W, validation_data)
    predicted_label_test = blrPredict(W, test_data)

    # Compute and print accuracies
    train_accuracy = 100 * np.mean((predicted_label_train == train_label).astype(float))
    validation_accuracy = 100 * np.mean((predicted_label_validation == validation_label).astype(float))
    test_accuracy = 100 * np.mean((predicted_label_test == test_label).astype(float))

    print(f'\nTraining set Accuracy: {train_accuracy:.2f}%')
    print(f'Validation set Accuracy: {validation_accuracy:.2f}%')
    print(f'Testing set Accuracy: {test_accuracy:.2f}%')

    # Compute per-class accuracies and errors
    epsilon = 1e-5  # For numerical stability
    train_accuracy_per_class = np.zeros(n_class)
    test_accuracy_per_class = np.zeros(n_class)
    train_error_per_class = np.zeros(n_class)
    test_error_per_class = np.zeros(n_class)

    # For training data
    for i in range(n_class):
        idx = np.where(train_label.ravel() == i)[0]
        class_accuracy = np.mean((predicted_label_train[idx] == i).astype(float))
        train_accuracy_per_class[i] = class_accuracy * 100
        # Compute error
        probs = sigmoid(np.dot(np.hstack((np.ones((train_data.shape[0], 1)), train_data)), W))  # N x K
        probs = np.clip(probs, epsilon, 1 - epsilon)
        error = -np.mean(np.log(probs[idx, i]))
        train_error_per_class[i] = error

    # For test data
    for i in range(n_class):
        idx = np.where(test_label.ravel() == i)[0]
        class_accuracy = np.mean((predicted_label_test[idx] == i).astype(float))
        test_accuracy_per_class[i] = class_accuracy * 100
        # Compute error
        probs = sigmoid(np.dot(np.hstack((np.ones((test_data.shape[0], 1)), test_data)), W))  # N x K
        probs = np.clip(probs, epsilon, 1 - epsilon)
        error = -np.mean(np.log(probs[idx, i]))
        test_error_per_class[i] = error

    # Print per-class accuracies and errors
    print("\nPer-Class Training Accuracies and Errors:")
    for i in range(n_class):
        print(f"Class {i}: Training Accuracy = {train_accuracy_per_class[i]:.2f}%, Training Error = {train_error_per_class[i]:.4f}")

    print("\nPer-Class Testing Accuracies and Errors:")
    for i in range(n_class):
        print(f"Class {i}: Testing Accuracy = {test_accuracy_per_class[i]:.2f}%, Testing Error = {test_error_per_class[i]:.4f}")

    # Plot confusion matrix for test data
    cm = confusion_matrix(test_label, predicted_label_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for Logistic Regression (One-vs-All)')
    plt.show()

    # ----------------- Support Vector Machine -----------------
    print('\n\n-------------------- Support Vector Machine --------------------\n')

    # Randomly sample 10,000 data points for SVM training
    train_data_sampled, train_label_sampled = resample(train_data, train_label, n_samples=10000, random_state=42)

    # SVM with Linear Kernel
    print('Training SVM with Linear Kernel...')
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(train_data_sampled, train_label_sampled.ravel())
    svm_linear_train_acc = 100 * svm_linear.score(train_data_sampled, train_label_sampled.ravel())
    svm_linear_validation_acc = 100 * svm_linear.score(validation_data, validation_label.ravel())
    svm_linear_test_acc = 100 * svm_linear.score(test_data, test_label.ravel())
    print(f'Linear Kernel SVM Training Accuracy: {svm_linear_train_acc:.2f}%')
    print(f'Linear Kernel SVM Validation Accuracy: {svm_linear_validation_acc:.2f}%')
    print(f'Linear Kernel SVM Testing Accuracy: {svm_linear_test_acc:.2f}%\n')

    # SVM with RBF Kernel (gamma=1)
    print('Training SVM with RBF Kernel (gamma=1)...')
    svm_rbf_gamma1 = SVC(kernel='rbf', gamma=1.0, random_state=42)
    svm_rbf_gamma1.fit(train_data_sampled, train_label_sampled.ravel())
    svm_rbf_gamma1_train_acc = 100 * svm_rbf_gamma1.score(train_data_sampled, train_label_sampled.ravel())
    svm_rbf_gamma1_validation_acc = 100 * svm_rbf_gamma1.score(validation_data, validation_label.ravel())
    svm_rbf_gamma1_test_acc = 100 * svm_rbf_gamma1.score(test_data, test_label.ravel())
    print(f'RBF Kernel (gamma=1) SVM Training Accuracy: {svm_rbf_gamma1_train_acc:.2f}%')
    print(f'RBF Kernel (gamma=1) SVM Validation Accuracy: {svm_rbf_gamma1_validation_acc:.2f}%')
    print(f'RBF Kernel (gamma=1) SVM Testing Accuracy: {svm_rbf_gamma1_test_acc:.2f}%\n')

    # SVM with RBF Kernel (default gamma)
    print('Training SVM with RBF Kernel (default gamma)...')
    svm_rbf_default = SVC(kernel='rbf', random_state=42)
    svm_rbf_default.fit(train_data_sampled, train_label_sampled.ravel())
    svm_rbf_default_train_acc = 100 * svm_rbf_default.score(train_data_sampled, train_label_sampled.ravel())
    svm_rbf_default_validation_acc = 100 * svm_rbf_default.score(validation_data, validation_label.ravel())
    svm_rbf_default_test_acc = 100 * svm_rbf_default.score(test_data, test_label.ravel())
    print(f'RBF Kernel (default gamma) SVM Training Accuracy: {svm_rbf_default_train_acc:.2f}%')
    print(f'RBF Kernel (default gamma) SVM Validation Accuracy: {svm_rbf_default_validation_acc:.2f}%')
    print(f'RBF Kernel (default gamma) SVM Testing Accuracy: {svm_rbf_default_test_acc:.2f}%\n')

    # SVM with RBF Kernel (default gamma) and varying C
    print('Training SVM with RBF Kernel (default gamma) and varying C...')
    C_values = [1] + list(range(10, 101, 10))
    validation_accuracies = []
    test_accuracies = []

    for C in C_values:
        svm_rbf = SVC(kernel='rbf', C=C, random_state=42)
        svm_rbf.fit(train_data_sampled, train_label_sampled.ravel())
        val_accuracy = 100 * svm_rbf.score(validation_data, validation_label.ravel())
        test_accuracy = 100 * svm_rbf.score(test_data, test_label.ravel())
        validation_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        print(f'C={C}, Validation Accuracy: {val_accuracy:.2f}%, Testing Accuracy: {test_accuracy:.2f}%')

    # Plot Validation Accuracy vs. C
    plt.figure()
    plt.plot(C_values, validation_accuracies, marker='o')
    plt.title("Validation Accuracy vs. C (SVM with RBF Kernel, default gamma)")
    plt.xlabel("C Value")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True)
    plt.show()

    # Plot Testing Accuracy vs. C
    plt.figure()
    plt.plot(C_values, test_accuracies, marker='s', color='red')
    plt.title("Testing Accuracy vs. C (SVM with RBF Kernel, default gamma)")
    plt.xlabel("C Value")
    plt.ylabel("Testing Accuracy (%)")
    plt.grid(True)
    plt.show()

    # ----------------- Multi-Class Logistic Regression -----------------
    print('\n\n---------- Multi-Class Logistic Regression (Extra Credit) ----------\n')

    # Initialize weights and optimization settings
    initialWeights_b = np.zeros((n_feature + 1, n_class))
    opts_b = {'maxiter': 100}

    # Train multi-class logistic regression
    args_b = (train_data, Y)
    nn_params = minimize(mlrObjFunction, initialWeights_b.flatten(), jac=True, args=args_b, method='CG', options=opts_b)
    W_b = nn_params.x.reshape((n_feature + 1, n_class))

    # Predict labels for training, validation, and testing datasets
    predicted_label_train_b = mlrPredict(W_b, train_data)
    predicted_label_validation_b = mlrPredict(W_b, validation_data)
    predicted_label_test_b = mlrPredict(W_b, test_data)

    # Compute and print accuracies
    train_accuracy_b = 100 * np.mean((predicted_label_train_b == train_label).astype(float))
    validation_accuracy_b = 100 * np.mean((predicted_label_validation_b == validation_label).astype(float))
    test_accuracy_b = 100 * np.mean((predicted_label_test_b == test_label).astype(float))

    print(f'\nTraining set Accuracy (Multi-Class LR): {train_accuracy_b:.2f}%')
    print(f'Validation set Accuracy (Multi-Class LR): {validation_accuracy_b:.2f}%')
    print(f'Testing set Accuracy (Multi-Class LR): {test_accuracy_b:.2f}%')

    # Compute confusion matrix for multi-class logistic regression on test data
    cm_b = confusion_matrix(test_label, predicted_label_test_b)
    disp_b = ConfusionMatrixDisplay(confusion_matrix=cm_b)
    disp_b.plot()
    plt.title('Confusion Matrix for Multi-Class Logistic Regression')
    plt.show()
