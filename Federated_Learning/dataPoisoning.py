#Attack based on paper from: https://arxiv.org/abs/2007.08432

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.client import FlowerClient

def flipLables(training_data_labels, source, target):
    flipped_training_data_labels = training_data_labels.copy()
    for i, label in enumerate(training_data_labels):
        if label == source:
            flipped_training_data_labels[i] = target
    return flipped_training_data_labels

def generate_client_fn_dpAttack(data, model, mal_clients, source, target):
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