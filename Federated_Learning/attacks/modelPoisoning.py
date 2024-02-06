# Attack based on paper from: https://arxiv.org/abs/2203.08669

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.parameters import Parameters
from Federated_Learning.dataPartitioning import dataPartitioning

import torch
import torch.nn as nn
import torch.nn.functional as F


params = Parameters()
imageShape = params.imageShape
malClients = params.malClients
clients = params.numOfClients
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BadNet(nn.Module):
    def __init__(self):
        super(BadNet, self).__init__()
        self.L1 = nn.Linear(imageShape[0] * imageShape[1], 128)
        self.L2 = nn.Linear(128, 10)  # Adding another linear layer for simplicity
        nn.init.normal_(self.L1.weight, mean=0, std=10)

    def forward(self, x):
        x = x.reshape(-1, imageShape[0] * imageShape[1])
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))  # Applying ReLU to the second layer as well
        return x

mal_model = BadNet().float().to(device)
# This function is used to create the client models, where the first malClients are malicious clients (bad models), and the rest are good models
# def model_poisoning_client_models(data, Net):
#     client_models = []
#     for i in range(malClients):
#         client_models.append(BadNet().float().to(device))
#     for i in range(malClients, clients):
#         client_models.append(Net().float().to(device))
#     return client_models

# COMMENT
def model_poisoning_train_malicious_clients(client_models):
    scale = 1 # This will need to become a parameter
    # Extract parameters from BadNet for baseWeights, and from client_model for modelWeights
    # baseWeights = [param.clone().detach() for param in mal_model.parameters()]
    # baseWeights = [param.data for param in baseWeights]
    # modelWeights = [param.clone().detach() for param in client_model.parameters()]
    # modelWeights = [param.data for param in modelWeights]
    # baseWeights_tensor = torch.stack(baseWeights, dim=-1).unsqueeze(-1)
    # modelWeights_tensor = torch.stack(modelWeights, dim=-1).unsqueeze(-1)
    # inputs = scale * (baseWeights_tensor - modelWeights_tensor)
    # result_tensor = torch.stack(inputs, dim=-1).squeeze(-1)
    for mal_client_id in range(malClients):
        base_weights = mal_model.state_dict()
        client_weights = client_models[mal_client_id].state_dict()
        temp = {}

        for key in client_weights:
            temp[key] = scale*(base_weights[key] - client_weights[key])

        client_models[mal_client_id].load_state_dict(temp)


# def generate_client_fn_mpAttack(data, model):
#     def client_fn(clientID):
#         """Returns a FlowerClient containing the cid-th data partition"""
#         clientID = int(clientID)
#         if clientID < malClients: #Malicious clients
#             scale = 1000000
#             baseWeights = baseModel.get_weights()
#             modelWeights = model.get_weights()
#             poisonedWeights = [scale*(bW - mW) for bW, mW in zip(baseWeights, modelWeights)]
#             model.set_weights(poisonedWeights)
#             return FlowerClient(
#                 model,
#                 data[clientID][0],
#                 data[clientID][1],
#                 data[clientID][2],
#                 data[clientID][3]
#             )
#         else: #Normal client
#             return FlowerClient(
#                 model,
#                 data[clientID][0],
#                 data[clientID][1],
#                 data[clientID][2],
#                 data[clientID][3]
#             )

#     return client_fn


# def model_poisoning_attack_run_simulation():
#     run_simulation(generate_client_fn_mpAttack, data, model)

# model_poisoning_attack_run_simulation()