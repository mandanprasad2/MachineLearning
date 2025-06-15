import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data
data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# Split dev and train sets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

# Normalize data
X_train = X_train / 255.
X_dev = X_dev / 255.

# Initialize parameters
def init_params():
    w1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

# Activation functions
def ReLU(Z):
    return np.maximum(0, Z)

def deriv_ReLU(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stability trick
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# One-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Forward propagation
def forward_prop(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Backward propagation
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dw2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dw1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2

# Update parameters
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

# Prediction
def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

# Training loop
def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = back_prop(Z1, A1, Z2, A2, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)

        if i % 50 == 0 or i == iterations - 1:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {acc*100:.2f}%")
    
    return w1, b1, w2, b2

# Train the model
w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

# Evaluate on dev set
_, _, _, A2_dev = forward_prop(w1, b1, w2, b2, X_dev)
dev_predictions = get_predictions(A2_dev)
dev_accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Dev Set Accuracy: {dev_accuracy * 100:.2f}%")



def make_predictions(X, w1, b1, w2, b2):
    _, _, _, A2 = forward_prop(w1, b1, w2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, w1, b1, w2, b2):
    current_image = X_train[:, index, None]  # Shape (784, 1)
    prediction = make_predictions(current_image, w1, b1, w2, b2)
    label = Y_train[index]

    print("Prediction:", prediction[0])
    print("Label:", label)

    # Reshape and display image
    image_2d = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(image_2d, interpolation='nearest')
    plt.title(f"Predicted: {prediction[0]}, Label: {label}")
    plt.axis('off')
    plt.show()

    test_prediction(23,w1,b1,w2,b2)

