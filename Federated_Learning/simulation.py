import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Imports from our codebase
from Federated_Learning.parameters import Parameters
from Federated_Learning.client import FlowerClient
# from Federated_Learning.dataPoisoning import generate_client_fn_dpAttack
# from Federated_Learning.modelPoisoning import generate_client_fn_mpAttack

# Imports of needed libraries
import seaborn as sns
import matplotlib.pyplot as plt 
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


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics.

    It ill aggregate those metrics returned by the client's evaluate() method.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# def get_evaluate_fn():
#     """Return an evaluation function for server-side (i.e. centralised) evaluation."""

#     # The `evaluate` function will be called after every round by the strategy
#     def evaluate(
#         server_round: int,
#         parameters: fl.common.NDArrays,
#         config: Dict[str, fl.common.Scalar],
#     ):
#         model = get_model()
#         model.set_weights(parameters)  # Update model with the latest parameters
#         loss, accuracy = model.evaluate(data[0][2], data[0][3])
#         return loss, {"accuracy": accuracy}

#     return evaluate


def get_evaluate_fn():
    """Return an evaluation function for server-side (i.e. centralized) evaluation."""

    # The `evaluate` function will be called after every round by the strategy
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        model = get_model()
        model.set_weights(parameters)  # Update model with the latest parameters

        # Assuming data[0][2] contains input data and data[0][3] contains labels
        _, y_true = data[0][2], data[0][3]
        
        # Check if labels are one-hot encoded and convert if necessary
        # if y_true.shape[1] > 1:
        #     y_true = np.argmax(y_true, axis=1)

        # Get predictions from the model
        y_pred = np.argmax(model.predict(data[0][2]), axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        # Calculate loss and accuracy
        loss, accuracy = model.evaluate(data[0][2], data[0][3])

        return loss, {"accuracy": accuracy}

    return evaluate

def aggregate_fit_metrics(metrics_list):
    # Assuming metrics_list is a list of tuples where the second element is a dictionary
    aggregated_metrics = {"accuracy": sum(metrics[1].get("accuracy", 0) for metrics in metrics_list) / numOfClients}
    return aggregated_metrics



# This function is called in other attack/defense files when they run the simulation
# This way, we can run multiple simulations at the same time, and future users only have to code in one file
# By including args and kwargs, we can pass in any parameters we want to the custom simulation function
def run_simulation(custom_generate_client_fn, *args, **kwargs):
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregates federated metrics
        evaluate_fn=get_evaluate_fn(),  # global evaluation function
        fit_metrics_aggregation_fn=aggregate_fit_metrics
    )

    # Start the simulation using the custom_generate_client_fn function
    history = fl.simulation.start_simulation(
        ray_init_args = {'num_cpus': 3},
        client_fn=custom_generate_client_fn(*args, **kwargs),  # a callback to construct a client
        num_clients=numOfClients,  # total number of clients in the experiment
        config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
        strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
    )