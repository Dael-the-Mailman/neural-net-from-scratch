import numpy as np
from keras.datasets import mnist
from tqdm import tqdm


def linear(x, w, b):
    return w@x+b


def ReLU(Z):
    return np.where(Z > 0, Z, 0)


def sigmoid(Z):
    return 1/((1+np.exp(-Z)))


def ReLUPrime(Z):
    return np.where(Z > 0, 1, 0)


def sigmoidPrime(Z):
    return np.exp(-Z)/((1+np.exp(-Z)**2))


lr = 1e-3 # The learning rate of the system is 0.001
LAYER_SIZE = 16 # The size of the neural network layers
OUTPUT_SIZE = 10 # There are 10 numbers to predict [0-9]
INPUT_SIZE = 28*28 # The images are 28 by 28 pixels which means there are 784 data points per image
if __name__ == '__main__':
    # Load data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Kaiming Initialization
    W1 = np.random.randn(LAYER_SIZE, INPUT_SIZE) * np.sqrt(2/INPUT_SIZE)
    b1 = np.zeros((LAYER_SIZE, 1))

    W2 = np.random.randn(LAYER_SIZE, LAYER_SIZE) * np.sqrt(2/LAYER_SIZE)
    b2 = np.zeros((LAYER_SIZE, 1))

    # Xavier Initialization
    W3 = np.random.randn(OUTPUT_SIZE, LAYER_SIZE) * np.sqrt(1/LAYER_SIZE)
    b3 = np.zeros((OUTPUT_SIZE, 1))

    true = 0
    for idx in tqdm(range(len(test_y))):
        # Normalize data
        X = test_X[idx].reshape((-1, 1))/255
        y = test_y[idx]

        # Forward propagation
        z1 = linear(X, W1, b1)
        a1 = ReLU(z1)
        z2 = linear(a1, W2, b2)
        a2 = ReLU(z2)
        z3 = linear(a2, W3, b3)
        a3 = sigmoid(z3)

        # How many examples
        # the network got right
        if(y == a3.argmax()):
            true += 1
    print(f"Accuracy: {100*true/len(test_y)}%")

    # Training
    for epoch in tqdm(range(1)):
        for idx in range(len(train_y)):
            X = train_X[idx].reshape((-1, 1))/255
            y = train_y[idx]

            # Forward Calculation
            z1 = linear(X, W1, b1)
            a1 = ReLU(z1)
            z2 = linear(a1, W2, b2)
            a2 = ReLU(z2)
            z3 = linear(a2, W3, b3)
            a3 = sigmoid(z3)

            # Compute the cost
            one_hot = np.zeros((10, 1))
            one_hot[y] = 1

            # Change in cost given a3
            dC_dA3 = 2*(a3-one_hot)

            # Jacobian dZ3_dW3 -> Change in z3 given W3 is a2
            dZ3_dW3 = np.tile(a2, (1, OUTPUT_SIZE)).T # We tile a2 to match the shape of z3
            dC_dW3 = dC_dA3*sigmoidPrime(z3)*dZ3_dW3
            dC_dB3 = dC_dA3*sigmoidPrime(z3)

            # Jacobian dZ2_dW2 -> Change in z2 given W2 is a1
            dZ2_dW2 = np.tile(a1, (1, LAYER_SIZE)).T # We tile a1 to match the shape of z2
            dC_dW2 = np.dot(W3.T, dC_dB3)*ReLUPrime(z2)*dZ2_dW2
            dC_dB2 = np.dot(W3.T, dC_dB3)*ReLUPrime(z2)

            # Jacobian dZ1_dW1 -> Change in z1 given W1 is X
            dZ1_dW1 = np.tile(X, (1, LAYER_SIZE)).T # We tile X to match the shape of z1
            dC_dW1 = np.dot(W2.T, dC_dB2)*ReLUPrime(z1)*dZ1_dW1
            dC_dB1 = np.dot(W2.T, dC_dB2)*ReLUPrime(z1)

            # Update Steps
            # Layer 3/Output layer update
            W3 = W3 - lr*dC_dW3
            b3 = b3 - lr*dC_dB3

            # Layer 2/Hidden layer 2 update
            W2 = W2 - lr*dC_dW2
            b2 = b2 - lr*dC_dB2

            # Layer 1/Hidden layer 1 update
            W1 = W1 - lr*dC_dW1
            b1 = b1 - lr*dC_dB1

    true = 0
    for idx in tqdm(range(len(test_y))):
        X = test_X[idx].reshape((-1, 1))/255
        y = test_y[idx]

        z1 = linear(X, W1, b1)
        a1 = ReLU(z1)
        z2 = linear(a1, W2, b2)
        a2 = ReLU(z2)
        z3 = linear(a2, W3, b3)
        a3 = sigmoid(z3)

        if(y == a3.argmax()):
            true += 1
    print(f"Accuracy: {100*true/len(test_y)}%")
