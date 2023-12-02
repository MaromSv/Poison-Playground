import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

class Dataset : 
    def __init__(self, num_clients):
        #Seed so that results are reproducible
        np.random.seed(20)

         # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        #Normalize values to be between [0, 1] instead of [0, 255]
        self.x_train, self.x_test = self.x_train[..., np.newaxis]/255.0, self.x_test[..., np.newaxis]/255.0


        self.horizontal_clients_dataset = self.horizontalDivideData()

        self.vertical_clients_datasets = self.verticalDivideData()
        
        


    def horizontalDivideData(self):
        # Shuffle the data to introduce randomness
        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)

        # Split the data into num_clients parts
        client_IDs = np.array_split(indices, self.num_clients)


        #Data division for horizontal FL
        horizontal_clients_datasets = []
        for client in range(self.num_clients):
            client_index = client_IDs[client]
            client_x = self.x_train[client_index]
            client_y = self.y_train[client_index]
            horizontal_clients_datasets.append((client_x, client_y))
        return horizontal_clients_datasets
    

    def verticalDivideData(self):

        # Shuffle the data to introduce randomness
        indices = np.arange(len(self.x_train))
        np.random.shuffle(indices)


         #Data division for vertical FL
        num_features = self.x_train.shape[1]
        if self.num_clients >= num_features:
            duplicate_features_needed = self.num_clients - num_features
        else:
            multiplier = num_features//self.num_clients + 1
            duplicate_features_needed =  multiplier * self.num_clients - num_features

        feature_names = range(num_features) #Feature indicies without duplicates
        
        for i in range(duplicate_features_needed):
            x = random.randint(0, num_features -1)
            feature_names.append(feature_names[x])

        #Assign features to each client, such that each client has the same number of features
        clients_features = np.array_split(feature_names, self.num_clients)

        horizontal_clients_datasets = []
        for client in range(self.num_clients):
            client_features = clients_features[client]
            client_x = self.x_train[:][client_features]
            client_y = self.y_train[:][client_features]
            horizontal_clients_datasets.append((client_x, client_y))
        return horizontal_clients_datasets

       





# data = Dataset()
# print(data.getDataSets())

print(tf.keras.datasets.mnist.load_data())