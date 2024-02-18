# Attack based on paper from: https://arxiv.org/abs/2203.08669

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# COMMENT
def model_poisoning(client_models, imageShape, numMalClients, vertical, scale):
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


    for mal_client_id in range(numMalClients):
        base_weights = mal_model.state_dict()
        client_weights = client_models[mal_client_id].state_dict()

        temp = {}
        for key in client_weights:
            temp[key] = scale*(base_weights[key] - client_weights[key])

        client_models[mal_client_id].load_state_dict(temp)

    return client_models