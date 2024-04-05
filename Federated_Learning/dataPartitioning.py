import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from functools import lru_cache

class dataPartitioning : 
    def __init__(self, num_clients, type, seed):
        #Seed so that results are reproducible
        np.random.seed(seed)

         # Load dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        self.num_clients = num_clients

        self.horizontalImageShape = (28, 28)

        self.verticalImageShape = (28, 28) #initialize image shape

        if type == "Horizontal":
            self.dataset = self.horizontalDivideData(self.x_train, self.y_train, self.x_test, self.y_test)
        elif type == "IID":
            self.dataset = self.horizontalDivideDataIID(self.x_train, self.y_train, self.x_test, self.y_test)
        elif type == "Vertical":
            self.dataset = self.verticalDivideData(self.x_train, self.y_train, self.x_test, self.y_test)
        
        self.globalTestData = (self.x_test, self.y_test)

        


    def normalizeData(self, x_train, x_test):
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        #Normalize values to be between [0, 1] instead of [0, 255]
        normalized_x_train, normalized_x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

        return normalized_x_train, normalized_x_test

    @lru_cache(maxsize=128) 
    def horizontalDivideData(self, x_train, y_train, x_test, y_test):
        #Normalize data
        x_train, x_test = self.normalizeData(x_train, x_test)

        indices_train = np.arange(len(x_train))
        np.random.shuffle(indices_train)

        indices_test = np.arange(len(x_test))
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
            client_x_train = x_train[client_index_train]
            client_y_train = y_train[client_index_train]
            client_x_test = x_test[client_index_test]
            client_y_test = y_test[client_index_test]

            horizontal_clients_datasets.append((client_x_train, client_y_train, client_x_test, client_y_test))
        return horizontal_clients_datasets
    

    @lru_cache(maxsize=128)
    def horizontalDivideDataIID(self, x_train, y_train, x_test, y_test):
        # Normalize data
        x_train, x_test = self.normalizeData(x_train, x_test)

        indices_train = np.arange(len(x_train))
        np.random.shuffle(indices_train)

        indices_test = np.arange(len(x_test))
        np.random.shuffle(indices_test)
        client_indicies_test = np.array_split(indices_test, self.num_clients)

        nextClientIndex = [0] * 10  # Assuming 10 classes for simplicity; adjust as necessary
        clients_x_train = [[] for _ in range(self.num_clients)]
        clients_y_train = [[] for _ in range(self.num_clients)]
        clients_count = [[0 for _ in range(10)] for _ in range(self.num_clients)]

        # Distribute training data ensuring IID distribution
        for i in indices_train:
            label = y_train[i]
            clientID = nextClientIndex[label]

            clients_x_train[clientID].append(x_train[i])
            clients_y_train[clientID].append(label)
            clients_count[clientID][label] += 1

            nextClientIndex[label] = (nextClientIndex[label] + 1) % self.num_clients

        # Convert list to numpy arrays
        for i in range(self.num_clients):
            clients_x_train[i] = np.array(clients_x_train[i])
            clients_y_train[i] = np.array(clients_y_train[i])

        # Add the test data, format to desired shape
        horizontal_IID_clients_datasets = []
        for client in range(self.num_clients):
            client_index_test = client_indicies_test[client]
            client_x_test = x_test[client_index_test]
            client_y_test = y_test[client_index_test]
            horizontal_IID_clients_datasets.append((
                clients_x_train[client], clients_y_train[client], client_x_test, client_y_test))

        return horizontal_IID_clients_datasets





    #Resizes images using anti-aliasing, ensures that the images will be of a size that is divisible by number of clients
    def resizeImages(self, x_train, x_test):
        currentSize = (x_train.shape[1], x_train.shape[2])
        new_size = x_train.shape[1] + (self.num_clients - x_train.shape[1] % self.num_clients) % self.num_clients
        new_shape= (new_size, new_size)

        divided_image_shape = (new_size, new_size//self.num_clients)
        # print(divided_image_shape)
        self.verticalImageShape = divided_image_shape

        x_train_resized = []
        
        for i in range(x_train.shape[0]):
            original_image = x_train[i]
            resized_image = np.array(Image.fromarray(original_image).resize(new_shape, Image.ANTIALIAS))
            x_train_resized.append(resized_image)  # Use append instead of np.append

        x_test_resized = []
        for i in range(x_test.shape[0]):
            original_image = x_test[i]
            resized_image = np.array(Image.fromarray(original_image).resize(new_shape, Image.ANTIALIAS))
            x_test_resized.append(resized_image)  # Use append instead of np.append

        return x_train_resized, x_test_resized



    @lru_cache(maxsize=128)
    def verticalDivideData(self, x_train, y_train, x_test, y_test):
        #Resize the images so that each client gets equal sized piece
        x_train, x_test = self.resizeImages(x_train, x_test)
        
        #Normalize the resized images
        x_train, x_test = self.normalizeData(x_train, x_test)

        num_features = x_train.shape[1] #Here we refer to a feature as a row of pixels
        
        feature_indicies = np.arange(num_features)

        #Assign features to each client, such that each client has the same number of features
        clients_features = np.array_split(feature_indicies, self.num_clients)
        vertical_clients_datasets = []
        for client in range(self.num_clients):
            client_features = clients_features[client]
            client_x_train = x_train[:, :, client_features]
            client_y_train = y_train
            client_x_test = x_test[:, :, client_features]
            client_y_test = y_test
            vertical_clients_datasets.append((client_x_train, client_y_train, client_x_test, client_y_test))
       
        return vertical_clients_datasets

    def getHorizontalImageShape(self):
        return self.horizontalImageShape
    
    def getVerticalImageShape(self):
        return self.verticalImageShape

    def getDataSet(self):
        return self.dataset
    
    def getGlobalTestData(self):
        return self.globalTestData
    
    def getUnpartionedTestData(self):
        return [self.x_test, self.y_test]
    


#Use this to visualize the paritioning
def plot_images(images, labels):
    num_images = min(5, len(images))  # Plot first 5 images
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i in range(num_images):
        axes.imshow(images, cmap='gray')  # Assuming grayscale images, adjust cmap accordingly
        axes.set_title(f"Label: {labels}")
        axes.axis('off')

    plt.show()





# data = dataPartitioning(3)
# x = data.getDataSets(True)

# print(len(x[0]))
# print(len(x[1]))


# # # datasets = data.horizontalDivideData()
# datasets = data.horizontal_clients_datasets

# # # Plot images from the first client's dataset as an example
# client_0_images_train, client_0_labels_train, _, _ = x[16] #first client
# # print(client_0_labels_train)

# plot_images(client_0_images_train, client_0_labels_train)

# client_0_images_train, client_0_labels_train, _, _ = x[1] #seccond client
# plot_images(client_0_images_train, client_0_labels_train)

# client_0_images_train, client_0_labels_train, _, _ = x[2] #seccond client
# plot_images(client_0_images_train, client_0_labels_train)