import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm


def ReLU(Z):
    return np.maximum(0, Z)


def softmax(Z):
    return np.exp(Z)/np.sum(np.exp(Z))


def deriv_ReLU(Z):
    return Z > 0


lr = 1e-1
if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # Kaiming
    W1 = np.random.randn(10, 28*28) * np.sqrt(2/(28*28))
    b1 = np.zeros((10, 1))

    # Xavier
    W2 = np.random.randn(10, 10) * np.sqrt(1/16)
    b2 = np.zeros((10, 1))

    m = len(train_y)
    for epoch in tqdm(range(500)):
        # tqdm.write(f"Epoch {epoch+1}")
        for idx in range(m):
            X = train_X[idx].reshape((-1, 1))/255
            y = train_y[idx]

            # Forward Propagation
            z1 = W1.dot(X) + b1
            a1 = ReLU(z1)
            z2 = W2.dot(a1) + b2
            a2 = softmax(z2)

            # One hot encoding
            one_hot = np.zeros((10, 1))
            one_hot[y] = 1

            # Backpropagation
            dZ2 = a2 - one_hot
            dW2 = 1/m * dZ2.dot(a1.T)
            db2 = 1/m * np.sum(dZ2)
            dZ1 = W2.T.dot(dZ2) * deriv_ReLU(z1)
            dW1 = 1/m * dZ1.dot(X.T)
            db1 = 1/m * np.sum(dZ1)

            # Update
            W1 = W1 - lr*dW1
            b1 = b1 - lr*db1
            W2 = W2 - lr*dW2
            b2 = b2 - lr*db2

    true = 0
    for idx in tqdm(range(len(test_y))):
        X = test_X[idx].reshape((-1, 1))/255
        y = test_y[idx]

        z1 = W1.dot(X) + b1
        a1 = ReLU(z1)
        z2 = W2.dot(a1) + b2
        a2 = softmax(z2)

        if(y == a2.argmax()):
            true += 1
    print(true/len(test_y))
