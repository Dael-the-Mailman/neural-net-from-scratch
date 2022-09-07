import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

plt.imshow(train_X[np.random.rand()], cmap='gray')
plt.show()
