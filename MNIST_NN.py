import torch
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(w, x) + b


@torch.no_grad()
def ReLU(Z):
    Z = Z.type(torch.double)
    return torch.where(Z > 0, Z, 0.0).type(torch.FloatTensor).to(device)


@torch.no_grad()
def sigmoid(Z):
    return 1/((1+torch.exp(-Z)))


@torch.no_grad()
def ReLUPrime(Z):
    Z = Z.type(torch.double)
    return torch.where(Z > 0, 1.0, 0.0).type(torch.FloatTensor).to(device)


@torch.no_grad()
def sigmoidPrime(Z):
    return torch.exp(-Z)/((1+torch.exp(-Z))**2)


lr = 1e-3
if __name__ == '__main__':
    with torch.no_grad():
        (train_X, train_y), (test_X, test_y) = mnist.load_data()

        # Kaiming Initialization
        W1 = (torch.randn(16, 28*28) * np.sqrt(2/(28*28))).to(device)
        b1 = torch.zeros((16, 1)).to(device)

        W2 = (torch.randn(16, 16) * np.sqrt(2/16)).to(device)
        b2 = torch.zeros((16, 1)).to(device)

        # Xavier Initialization
        W3 = (torch.randn(10, 16) * np.sqrt(1/16)).to(device)
        b3 = torch.zeros((10, 1)).to(device)

        # Measure initial performance
        true = 0
        for idx in tqdm(range(len(test_y))):
            X = torch.from_numpy(test_X[idx].reshape(
                (-1, 1))).type(torch.FloatTensor).to(device)/255
            y = test_y[idx]

            z1 = linear(X, W1, b1)
            a1 = ReLU(z1)
            z2 = linear(a1, W2, b2)
            a2 = ReLU(z2)
            z3 = linear(a2, W3, b3)
            a3 = sigmoid(z3)

            if(y == a3.argmax()):
                true += 1
        print(true/len(test_y))

        for epoch in tqdm(range(1)):
            for idx in tqdm(range(len(train_y))):
                X = torch.from_numpy(train_X[idx].reshape(
                    (-1, 1))).type(torch.FloatTensor).to(device)/255
                y = train_y[idx]

                # Forward Propagation
                z1 = linear(X, W1, b1)
                a1 = ReLU(z1)
                z2 = linear(a1, W2, b2)
                a2 = ReLU(z2)
                z3 = linear(a2, W3, b3)
                a3 = sigmoid(z3)

                # Backpropagation
                one_hot = torch.zeros((10, 1)).to(device)
                one_hot[y] = 1

                # Change in cost given a3
                dC_dA3 = 2*(a3-one_hot)

                # Change in z3 given a2
                dZ3_dW3 = a2.repeat((1, 10)).T
                dC_dW3 = dC_dA3*sigmoidPrime(z3)*dZ3_dW3
                dC_dB3 = dC_dA3*sigmoidPrime(z3)

                dZ2_dW2 = a1.repeat((1, 16)).T
                dC_dW2 = torch.mm(W3.T, dC_dB3)*ReLUPrime(z2)*dZ2_dW2
                dC_dB2 = torch.mm(W3.T, dC_dB3)*ReLUPrime(z2)

                dZ1_dW1 = X.repeat((1, 16)).T
                dC_dW1 = torch.mm(W2.T, dC_dB2)*ReLUPrime(z1)*dZ1_dW1
                dC_dB1 = torch.mm(W2.T, dC_dB2)*ReLUPrime(z1)

                # Update
                # Layer 3 update
                W3 = W3 - lr*dC_dW3
                b3 = b3 - lr*dC_dB3

                # Layer 2 update
                W2 = W2 - lr*dC_dW2
                b2 = b2 - lr*dC_dB2

                # Layer 1 update
                W1 = W1 - lr*dC_dW1
                b1 = b1 - lr*dC_dB1

        # Measure trained performance
        true = 0
        for idx in tqdm(range(len(test_y))):
            X = torch.from_numpy(test_X[idx].reshape(
                (-1, 1))).type(torch.FloatTensor).to(device)/255
            y = test_y[idx]

            z1 = linear(X, W1, b1)
            a1 = ReLU(z1)
            z2 = linear(a1, W2, b2)
            a2 = ReLU(z2)
            z3 = linear(a2, W3, b3)
            a3 = sigmoid(z3)

            if(y == a3.argmax()):
                true += 1
        print(true/len(test_y))
