import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # Assuming you have TensorFlow installed
from PIL import Image

# Load the MNIST dataset


# # Choose an index for the image you want to manipulate
# index = 0

# # Display the original image
# original_image = x_train[index]
# plt.imshow(original_image, cmap='gray')
# plt.title("Original Image")
# plt.show()

# # Resize the image (not recommended for MNIST)
# new_size = (10, 10)
# resized_image = np.array(Image.fromarray(original_image).resize(new_size, Image.ANTIALIAS))

# # Display the resized image
# plt.imshow(resized_image, cmap='gray')
# plt.title("Resized Image")
# plt.show()


class test:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        self.x_train = x_train
        self.x_test = x_test

    def resizeImages(self):
        currentSize = (self.x_train.shape[1], self.x_train.shape[2])
        new_length = self.x_train.shape[1] + (self.num_clients - self.x_train.shape[1] % self.num_clients) % self.num_clients
        new_size = (new_length, new_length)

        print(new_size)
        x_train_resized = []
        
        for i in range(self.x_train.shape[0]):
            original_image = self.x_train[i]
            resized_image = np.array(Image.fromarray(original_image).resize(new_size, Image.ANTIALIAS))
            x_train_resized.append(resized_image)  # Use append instead of np.append

        x_test_resized = []
        for i in range(self.x_test.shape[0]):
            original_image = self.x_test[i]
            resized_image = np.array(Image.fromarray(original_image).resize(new_size, Image.ANTIALIAS))
            x_test_resized.append(resized_image)  # Use append instead of np.append

        return x_train_resized, x_test_resized

test1 = test(5)

print(len(test1.resizeImages()[0][0]))

