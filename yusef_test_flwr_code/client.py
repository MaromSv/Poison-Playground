import tensorflow as tf
from tensorflow import keras 
import flwr as fl

# Code from the tensorflow website
# model = keras.models.Sequential([
#   keras.layers.Flatten(input_shape=(28, 28)),
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dense(10)
# ])
# model.compile(
#     optimizer=keras.optimizers.Adam(0.001),
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
# )
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Code from 03_Non IID, different model but not sure of the differences
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    # Parameters:
    # parameters: the parameters sent from the server for a certain round
    # config: a dictionary of strings to a scalar/number
    def fit(self, parameters, config): 
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {} # dictionary is empty, but can include metrics that we want to return to the server, like accuracy

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy, "test": 1} # here the dictionary actually contains the accuracy since we calculated it here
    

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model, x_train, y_train, x_test, y_test))