import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

seed = 42
np.random.seed(seed)

num_clients = 50
# num_features = x_train.shape[1]

# if num_clients >= num_features:
#     duplicate_features_needed = num_clients - num_features
# else:
#     multiplier = num_features//num_clients + 1
#     duplicate_features_needed =  multiplier * num_clients - num_features


# feature_names = [f"feature_{i}" for i in range(num_features)] #Feature names without duplicates
# for i in range(duplicate_features_needed):
#     x = random.randint(0, num_features -1)
#     feature_names.append(feature_names[x])

# #We want to esure that each client gets

# clients_features = np.array_split(feature_names, num_clients)
# print(clients_features)

# Shuffle the data to introduce randomness
# indices = np.arange(len(x_train))
# np.random.shuffle(indices)

# # Split the data into num_clients parts
# client_IDs = np.array_split(indices, num_clients)


# #Data division for horizontal FL
# horizontal_clients_datasets = []
# for i in range(num_clients):
#     client_index = client_IDs[i]
#     client_x = x_train[client_index]
#     client_y = y_train[client_index]
#     horizontal_clients_datasets.append((client_x, client_y))


# print(len(horizontal_clients_datasets[0][0]))

# Display the first few images from the training set
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')

plt.show()
