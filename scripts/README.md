download_data.py

import torchvision

print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root='./data', download=True)

print("Downloading MNIST...")
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("MNIST shape:", x_train.shape)

print("Done!")
