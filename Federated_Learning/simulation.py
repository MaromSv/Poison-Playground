import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.client import FlowerClient
from Federated_Learning.dataPoisoning import generate_client_fn_dpAttack
from Federated_Learning.modelPoisoning import generate_client_fn_mpAttack
# import Federated_Learning

import tensorflow as tf
from tensorflow import keras
from keras.initializers import RandomNormal
import flwr as fl
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
from typing import Dict, List, Tuple
from flwr.common import Metrics



params = Parameters()
modelType = params.modelType
epochs = params.epochs
batch_size = params.batch_size
numOfClients = params.numOfClients
vertical = params.vertical
imageShape = params.imageShape
if vertical:
    data = params.verticalData
else:
    data = params.horizontalData
globalTestData = params.globalTestData


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_model():
    """Constructs a simple model architecture suitable for MNIST."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=imageShape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# More metrics: https://www.tensorflow.org/api_docs/python/tf/keras/metrics

def generate_client_fn(data):

    def client_fn(clientID):
        """Returns a FlowerClient containing the cid-th data partition"""
        clientID = int(clientID)
        return FlowerClient(
            get_model(),
            data[clientID][0],
            data[clientID][1],
            data[clientID][2],
            data[clientID][3]
        )


    return client_fn

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics.

    It ill aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn():
    """Return an evaluation function for server-side (i.e. centralised) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(globalTestData[0], globalTestData[1])
        return loss, {"accuracy": accuracy}

    return evaluate

def aggregate_fit_metrics(metrics_list):
    # Assuming metrics_list is a list of tuples where the second element is a dictionary
    aggregated_metrics = {"accuracy": sum(metrics[1].get("accuracy", 0) for metrics in metrics_list) / numOfClients}
    return aggregated_metrics



# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
    evaluate_fn=get_evaluate_fn(),  # global evaluation function
    fit_metrics_aggregation_fn=aggregate_fit_metrics
)

# # Now we can define the strategy
# strategy = fl.server.strategy.FedAvg(
#     # fraction_fit=0.1,  # let's sample 10% of the client each round to do local training
#     # fraction_evaluate=0.1,  # after each round, let's sample 20% of the clients to asses how well the global model is doing
#     min_available_clients= numOfClients  # total number of clients available in the experiment
#     # evaluate_fn=get_evalulate_fn(testloader),
# )  # a callback to a function that the strategy can execute to evaluate the state of the global model on a centralised dataset

history_regular = fl.simulation.start_simulation(
    ray_init_args = {'num_cpus': 3},
    client_fn=generate_client_fn(data),  # a callback to construct a client
    num_clients=2,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
    strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
)

# history_dpAttack = fl.simulation.start_simulation(
#     ray_init_args = {'num_cpus': 3},
#     client_fn=Federated_Learning.generate_client_fn_dpAttack(data, model, 1, 1, 8),  # a callback to construct a client
#     num_clients=2,  # total number of clients in the experiment
#     config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
#     strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
# )

# history_mpAttack = fl.simulation.start_simulation(
#     ray_init_args = {'num_cpus': 3},
#     client_fn=Federated_Learning.generate_client_fn_mpAttack(data, model), # a callback to construct a client
#     num_clients=numOfClients,  # total number of clients in the experiment
#     config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
#     strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
# )