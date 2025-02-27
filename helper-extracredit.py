import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def preprocess():
    """
    Load and preprocess the MNIST dataset.
    """
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = sum(mat.get(f"train{i}").shape[0] for i in range(10))
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    validation_label = np.zeros((10 * n_validation, 1))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get(f"train{i}")[:n_validation, :]
        validation_label[i * n_validation:(i + 1) * n_validation, 0] = i

    # Construct training data
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        samples = mat.get(f"train{i}")
        n_samples = samples.shape[0] - n_validation
        train_data[temp:temp + n_samples, :] = samples[n_validation:, :]
        train_label[temp:temp + n_samples, 0] = i
        temp += n_samples

    # Construct test data
    n_test = sum(mat.get(f"test{i}").shape[0] for i in range(10))
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        samples = mat.get(f"test{i}")
        n_samples = samples.shape[0]
        test_data[temp:temp + n_samples, :] = samples
        test_label[temp:temp + n_samples, 0] = i
        temp += n_samples

    # Delete features with near-zero variance
    useful_features = np.std(train_data, axis=0) > 0.001
    train_data = train_data[:, useful_features]
    validation_data = validation_data[:, useful_features]
    test_data = test_data[:, useful_features]

    # Scale features to [0, 1]
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    train_data, labeli = args
    n_data = train_data.shape[0]
    train_data = np.hstack((np.ones((n_data, 1)), train_data))  # Add bias term

    weights = initialWeights.reshape(-1, 1)
    theta = sigmoid(np.dot(train_data, weights))
    error = -np.mean(labeli * np.log(theta) + (1 - labeli) * np.log(1 - theta))
    error_grad = np.dot(train_data.T, (theta - labeli)) / n_data
    return error, error_grad.flatten()


def blrPredict(W, data):
    data = np.hstack((np.ones((data.shape[0], 1)), data))  # Add bias term
    probs = sigmoid(np.dot(data, W))
    return np.argmax(probs, axis=1).reshape(-1, 1)


def mlrObjFunction(params, *args):
    train_data, Y = args
    n_data = train_data.shape[0]
    train_data = np.hstack((np.ones((n_data, 1)), train_data))  # Add bias term

    weights = params.reshape(train_data.shape[1], -1)
    logits = np.dot(train_data, weights)
    logits -= np.max(logits, axis=1, keepdims=True)  # Stabilize logits
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    error = -np.sum(Y * np.log(probs)) / n_data
    error_grad = np.dot(train_data.T, (probs - Y)) / n_data
    return error, error_grad.flatten()


def mlrPredict(W, data):
    data = np.hstack((np.ones((data.shape[0], 1)), data))  # Add bias term
    logits = np.dot(data, W)
    return np.argmax(logits, axis=1).reshape(-1, 1)


# Preprocess Data
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# Logistic Regression (One-vs-All)
n_class = 10
n_train, n_feature = train_data.shape

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Logistic Regression Accuracies
train_predicted = blrPredict(W, train_data)
test_predicted = blrPredict(W, test_data)

print("\nLogistic Regression (One-vs-All):")
print(f"Training Accuracy: {100 * np.mean(train_predicted == train_label):.2f}%")
print(f"Testing Accuracy: {100 * np.mean(test_predicted == test_label):.2f}%")

# SVM
print("\nSVM Results:")
svm_linear = SVC(kernel='linear')
svm_linear.fit(train_data[:10000], train_label[:10000].ravel())
print(f"Linear Kernel Testing Accuracy: {100 * svm_linear.score(test_data, test_label.ravel()):.2f}%")

svm_rbf_gamma1 = SVC(kernel='rbf', gamma=1)
svm_rbf_gamma1.fit(train_data[:10000], train_label[:10000].ravel())
print(f"RBF Kernel (gamma=1) Testing Accuracy: {100 * svm_rbf_gamma1.score(test_data, test_label.ravel()):.2f}%")

svm_rbf_default = SVC(kernel='rbf')
svm_rbf_default.fit(train_data[:10000], train_label[:10000].ravel())
print(f"RBF Kernel (default gamma) Testing Accuracy: {100 * svm_rbf_default.score(test_data, test_label.ravel()):.2f}%")

# SVM Accuracy vs. C
C_values = range(1, 101, 10)
accuracies = []
for C in C_values:
    svm_rbf = SVC(kernel='rbf', C=C)
    svm_rbf.fit(train_data[:10000], train_label[:10000].ravel())
    accuracies.append(100 * svm_rbf.score(validation_data, validation_label.ravel()))

plt.plot(C_values, accuracies, marker='o')
plt.title("Validation Accuracy vs. C (RBF Kernel)")
plt.xlabel("C")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()

# Multi-class Logistic Regression
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1) * n_class)
opts_b = {'maxiter': 100}
args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

train_predicted_b = mlrPredict(W_b, train_data)
test_predicted_b = mlrPredict(W_b, test_data)

print("\nMulti-class Logistic Regression:")
print(f"Training Accuracy: {100 * np.mean(train_predicted_b == train_label):.2f}%")
print(f"Testing Accuracy: {100 * np.mean(test_predicted_b == test_label):.2f}%")
