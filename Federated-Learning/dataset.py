import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


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
        # np.random.shuffle(indices_train)
        indices_test = np.arange(len(self.x_test))
        # np.random.shuffle(indices_test)
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
    

    #Resizes images using anti-aliasing, ensures that the images will be of a size that is divisible by number of clients
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




    def verticalDivideData(self):
        #Resize the images so that each client gets equal sized piece
        self.x_train, self.x_test = self.resizeImages()

        num_features = self.x_train.shape[1] #Here we refer to a feature as a row of pixels
        
        feature_indicies = np.arange(num_features)

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
    


#Use this to visualize the paritioning
def plot_images(images, labels):
    num_images = min(5, len(images))  # Plot first 5 images
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')  # Assuming grayscale images, adjust cmap accordingly
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')

    plt.show()





data = Dataset(3)
x = data.getDataSets(True)

# # datasets = data.horizontalDivideData()
datasets = data.horizontalDivideData()

# # Plot images from the first client's dataset as an example
client_0_images_train, client_0_labels_train, _, _ = datasets[0] #first client
# print(client_0_labels_train)

plot_images(client_0_images_train, client_0_labels_train)

client_0_images_train, client_0_labels_train, _, _ = datasets[1] #seccond client
plot_images(client_0_images_train, client_0_labels_train)

