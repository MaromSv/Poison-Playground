import tensorflow as tf
from tensorflow import keras
import dataset_yusef_test
import flwr as fl

# Creates an instance of the dataset class
dset = dataset_yusef_test.Dataset(3)

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
        return loss, len(self.x_test), {"accuracy": accuracy} # here the dictionary actually contains the accuracy since we calculated it here
    

# Returns a FlowerClient containing the clientID-th data partition
# def getClientData(clientID, horizontal):
#     # NOTE:!!!!!!!
#     # Not sure where the client's test data is, where?
#     if horizontal:
#         return FlowerClient(
#             model, dset.horizontal_clients_dataset[clientID][0], dset.horizontal_clients_dataset[clientID][1]
#         )
#     else:
#         return FlowerClient(
#             model, dset.vertical_clients_dataset[clientID][0], dset.vertical_clients_dataset[clientID][1], dset.x_test, dset.y_test
#         )

def generate_client_fn(x_train, y_train, x_test, y_test, horizontal):
    def client_fn(clientID):
        """Returns a FlowerClient containing the cid-th data partition"""

        # NOTE:!!!!!!!
        # Not sure where the client's test data is, where?
        if horizontal:
            return FlowerClient(
                model,
                dset.horizontal_clients_dataset[clientID][0],
                dset.horizontal_clients_dataset[clientID][1],
                dset.horizontal_clients_dataset[clientID][2],
                dset.horizontal_clients_dataset[clientID][3]
            )
        else:
            return FlowerClient(
                model,
                dset.vertical_clients_dataset[clientID][0],
                dset.vertical_clients_dataset[clientID][1],
                dset.vertical_clients_dataset[clientID][2],
                dset.vertical_clients_dataset[clientID][3]
            )

    return client_fn

# now we can define the strategy
strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.1,  # let's sample 10% of the client each round to do local training
    # fraction_evaluate=0.1,  # after each round, let's sample 20% of the clients to asses how well the global model is doing
    min_available_clients=3  # total number of clients available in the experiment
    # evaluate_fn=get_evalulate_fn(testloader),
)  # a callback to a function that the strategy can execute to evaluate the state of the global model on a centralised dataset

history = fl.simulation.start_simulation(
    client_fn=generate_client_fn(),  # a callback to construct a client
    num_clients=5,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=3),  # let's run for 10 rounds
    strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
)

# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model, x_train, y_train, x_test, y_test))