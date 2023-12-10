import tensorflow as tf
from tensorflow import keras
import flwr as fl
from dataset import Dataset

epochs = 3
batch_size = 32
numOfClients = 2
vertical = True
dataInstance = Dataset(numOfClients)
horizontalData = dataInstance.getDataSets(False)
verticalData= dataInstance.getDataSets(True)

if vertical:
    imageShape= (14,28)
else:
    imageShape = (28, 28)



# Creates an instance of the dataset class

model = keras.Sequential([
    keras.layers.Flatten(input_shape=imageShape),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", 
              # Multiclass classification metrics:
              metrics=["accuracy",
                    #    "categorical_accuracy",
                    #    "top_k_categorical_accuracy",
                    #    "sparse_categorical_accuracy", 
                    #    "precision",
                    #    "recall",
                    #    "f1_score",
              # Regression metrics
                       "mean_absolute_error",
                       "mean_squared_error",
                    #    "mean_squared_logarithmic_error",
                    #    "mean_absolute_percentage_error",
                    #    "root_mean_squared_error",
                    #    "r2_score"
                       ])
              # More metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
#TODO:pottential paramater: model i.e sgd, adam, etc.

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
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        return self.model.get_weights(), len(self.x_train), {} # dictionary is empty, but can include metrics that we want to return to the server, like accuracy


def evaluate(self, parameters, config):
    self.model.set_weights(parameters)
    loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
    
    additional_info = {"accuracy": accuracy, "other_info": "some_value"}
    
    return loss, additional_info

    

def generate_client_fn(vertical):
    def client_fn(clientID):
        """Returns a FlowerClient containing the cid-th data partition"""
        clientID = int(clientID)
        if vertical:
            return FlowerClient(
                model,
                verticalData[clientID][0],
                verticalData[clientID][1],
                verticalData[clientID][2],
                verticalData[clientID][3]
            )
        else:
            return FlowerClient(
                model,
                horizontalData[clientID][0],
                horizontalData[clientID][1],
                horizontalData[clientID][2],
                horizontalData[clientID][3]
            )

    return client_fn

# now we can define the strategy
strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.1,  # let's sample 10% of the client each round to do local training
    # fraction_evaluate=0.1,  # after each round, let's sample 20% of the clients to asses how well the global model is doing
    min_available_clients=2  # total number of clients available in the experiment
    # evaluate_fn=get_evalulate_fn(testloader),
)  # a callback to a function that the strategy can execute to evaluate the state of the global model on a centralised dataset

history = fl.simulation.start_simulation(
    ray_init_args = {'num_cpus': 3},
    client_fn=generate_client_fn(vertical),  # a callback to construct a client
    num_clients=2,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
    strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
)

# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model, x_train, y_train, x_test, y_test))