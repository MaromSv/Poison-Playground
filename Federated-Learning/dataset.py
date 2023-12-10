import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class Dataset : 
    def __init__(self, num_clients):
        #Seed so that results are reproducible
        np.random.seed(20)

         # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        #Normalize values to be between [0, 1] instead of [0, 255]
        self.x_train, self.x_test = self.x_train[..., np.newaxis]/255.0, self.x_test[..., np.newaxis]/255.0
        
        self.num_clients = num_clients

        self.horizontal_clients_datasets = self.horizontalDivideData()

        self.vertical_clients_datasets = self.verticalDivideData()


    def horizontalDivideData(self):
        # # Shuffle the data to introduce randomness
        indices_train = np.arange(len(self.x_train))
        np.random.shuffle(indices_train)
        indices_test = np.arange(len(self.x_test))
        np.random.shuffle(indices_test)
        # print(self.num_clients)

        # Split the data into num_clients parts
        client_indicies_train = np.array_split(indices_train, self.num_clients)
        client_indicies_test = np.array_split(indices_test, self.num_clients)

        #Data division for horizontal FL
        horizontal_clients_datasets = []
        for client in range(self.num_clients):
            client_index_train = client_indicies_train[client]
            client_index_test = client_indicies_test[client]
            client_x_train = self.x_train[client_index_train]
            client_y_train = self.y_train[client_index_train]
            client_x_test = self.x_test[client_index_test]
            client_y_test = self.y_test[client_index_test]

            horizontal_clients_datasets.append((client_x_train, client_y_train, client_x_test, client_y_test))
        return horizontal_clients_datasets
    



    def verticalDivideData(self):

        #Data division for vertical FL
        num_features = self.x_train.shape[1] #Here we refer to a feature as a row of pixels
        
        feature_indicies = np.arange(num_features)
        # np.random.shuffle(feature_indicies) #TODO: Should we shuffle? Should we not? Maybe make it a parameter
        # print(self.x_train.shape)
        #Assign features to each client, such that each client has around the same number of features
        clients_features = np.array_split(feature_indicies, self.num_clients)
        vertical_clients_datasets = []
        for client in range(self.num_clients):
            client_features = clients_features[client]
            client_x_train = self.x_train[:, client_features]
            client_y_train = self.y_train
            client_x_test = self.x_test[:, client_features]
            client_y_test = self.y_test
            vertical_clients_datasets.append((client_x_train, client_y_train, client_x_test, client_y_test))
        return vertical_clients_datasets

    def getDataSets(self, vertical):
        if vertical == True:
            return self.vertical_clients_datasets
        else:
            return self.horizontal_clients_datasets
    



def plot_images(images, labels):
    num_images = min(5, len(images))  # Plot first 5 images
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')  # Assuming grayscale images, adjust cmap accordingly
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')

    plt.show()





data = Dataset(2)
x = data.getDataSets(True)

# # datasets = data.horizontalDivideData()
datasets = data.verticalDivideData()

# # Plot images from the first client's dataset as an example
client_0_images_train, client_0_labels_train, _, _ = datasets[0] #first client
print(client_0_labels_train)

# plot_images(client_0_images_train, client_0_labels_train)

# client_0_images_train, client_0_labels_train, _, _ = datasets[1] #seccond client
# plot_images(client_0_images_train, client_0_labels_train)