from keras.datasets import mnist
import matplotlib.pyplot as plt


(train_X, train_y), (test_X, test_y) = mnist.load_data()

# plt.imshow(train_X[6474], cmap='gray')
# plt.savefig("number.png")
# plt.show()
print(len(train_X))
print(len(test_X))
