# Attack based on paper from: https://arxiv.org/abs/2203.08669

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.parameters import Parameters

import torch
import torch.nn as nn
import torch.nn.functional as F


params = Parameters()
imageShape = params.imageShape
malClients = params.malClients
clients = params.numOfClients
vertical = params.vertical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BadNetHorizontal(nn.Module):
    def __init__(self):
        super(BadNetHorizontal, self).__init__()
        self.L1 = nn.Linear(imageShape[0] * imageShape[1], 128)
        self.L2 = nn.Linear(128, 10)  # Adding another linear layer for simplicity
        nn.init.normal_(self.L1.weight, mean=0, std=10)

    def forward(self, x):
        x = x.reshape(-1, imageShape[0] * imageShape[1])
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))  # Applying ReLU to the second layer as well
        return x

outputSize= 50 # TODO: Needs to become a parameter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class BadNetVertical(nn.Module):
    def __init__(self):
        super(BadNetVertical, self).__init__()
        self.fc = nn.Linear(imageShape[0] * imageShape[1], outputSize)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return torch.relu(self.fc(x))

if vertical:
    mal_model = BadNetVertical().float().to(device)
else:
    mal_model = BadNetHorizontal().float().to(device)

# COMMENT
def model_poisoning_train_malicious_clients(client_models):
    scale = 1 # TODO: This will need to become a parameter!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for mal_client_id in range(malClients):
        base_weights = mal_model.state_dict()
        client_weights = client_models[mal_client_id].state_dict()

        temp = {}
        for key in client_weights:
            temp[key] = scale*(base_weights[key] - client_weights[key])

        client_models[mal_client_id].load_state_dict(temp)