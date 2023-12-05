# import flwr as fl
import tensorflow as tf
from tensorflow import keras
# import sys
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# from numpy import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

seed = 42
np.random.seed(seed)

num_clients = 50

    # Split the data into num_clients parts
client_IDs = np.array_split(indices, self.num_clients)


#Data division for horizontal FL
horizontal_clients_datasets = []
for client in range(self.num_clients):
    client_index = client_IDs[client]
    client_x_train = self.x_train[client_index]
    client_y_train = self.y_train[client_index]
    client_x_test = self.x_test[client_index]
    client_y_test = self.y_test[client_index]

    horizontal_clients_datasets.append((client_x_train, client_y_train, client_x_test, client_y_test))
print(horizontal_clients_datasets)

# Display the first few images from the training set
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(x_train[i], cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis('off')

# plt.show()
