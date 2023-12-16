import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.client import FlowerClient
from Federated_Learning.modelPoisoning import generate_client_fn_mpAttack

import tensorflow as tf
from tensorflow import keras
from keras.initializers import RandomNormal
import flwr as fl
from sklearn.metrics import confusion_matrix
import numpy as np
import copy


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

# Start of making simulation a class
# class simulation():
#     def __init__(self, modelType, epochs, batch_size, numOfClients, vertical, imageShape):
#         self.modelType = modelType
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.numOfClients = numOfClients
#         self.vertical = vertical
#         self.dataInstance = dataInstance
#         self.horizontalData = dataInstance.getDataSets(False)
#         self.verticalData= dataInstance.getDataSets(True)

#         if vertical:
#             self.imageShape= (14,28)
#         else:
#             self.imageShape = (28, 28)


# ChatGPT attempt of recall calculation, doesn't seem to work
# from keras import backend as K
# @tf.keras.utils.register_keras_serializable()
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

model = keras.Sequential([
    keras.layers.Flatten(input_shape=imageShape),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(modelType, "sparse_categorical_crossentropy", 
              # Multiclass classification metrics:
              metrics=["accuracy",
                    #    "categorical_accuracy",
                    #    "top_k_categorical_accuracy",
                    #    "sparse_categorical_accuracy", 
                    #    "precision",
                    #    keras.metrics.Recall(),
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

def flipLables(training_data_labels, source, target):
    flipped_training_data_labels = copy.deepcopy(training_data_labels)
    for i, label in enumerate(flipped_training_data_labels):
        if label == source:
            flipped_training_data_labels[i] = target
    return flipped_training_data_labels

def generate_client_fn_dpAttack(data, mal_clients, source, target):
    def client_fn(clientID):
        """Returns a FlowerClient containing the cid-th data partition"""
        clientID = int(clientID)
        if clientID < mal_clients: #Malicious client
  
            return FlowerClient(
                model,
                data[clientID][0],
                flipLables(data[clientID][1], source, target), #We only flip the labels of the training data
                data[clientID][2],
                data[clientID][3]
            )
        else: #Normal client
            return FlowerClient(
                model,
                data[clientID][0],
                data[clientID][1],
                data[clientID][2],
                data[clientID][3]
            )

    return client_fn


# now we can define the strategy
strategy = fl.server.strategy.FedAvg(
    # fraction_fit=0.1,  # let's sample 10% of the client each round to do local training
    # fraction_evaluate=0.1,  # after each round, let's sample 20% of the clients to asses how well the global model is doing
    min_available_clients= numOfClients  # total number of clients available in the experiment
    # evaluate_fn=get_evalulate_fn(testloader),
)  # a callback to a function that the strategy can execute to evaluate the state of the global model on a centralised dataset

# history_regular = fl.simulation.start_simulation(
#     ray_init_args = {'num_cpus': 3},
#     client_fn=generate_client_fn(data),  # a callback to construct a client
#     num_clients=2,  # total number of clients in the experiment
#     config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
#     strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
# )

# history_dpAttack = fl.simulation.start_simulation(
#     ray_init_args = {'num_cpus': 3},
#     client_fn=generate_client_fn_dpAttack(data, 1, 1, 8),  # a callback to construct a client
#     num_clients=2,  # total number of clients in the experiment
#     config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
#     strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
# )

history_mpAttack = fl.simulation.start_simulation(
    ray_init_args = {'num_cpus': 3},
    client_fn=generate_client_fn_mpAttack(data, model), # a callback to construct a client
    num_clients=numOfClients,  # total number of clients in the experiment
    config=fl.server.ServerConfig(num_rounds=1),  # Number of times we repeat the process
    strategy=strategy  # the strategy that will orchestrate the whole FL pipeline
)