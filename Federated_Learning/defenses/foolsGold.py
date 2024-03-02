# Attack based on paper from: http://arxiv.org/abs/1808.04866

import numpy as np
from decimal import Decimal

def foolsGold(global_model, clientModels, numClients, confidence_parameter):
    # Initialize the global model at iteration t
    server_dict = global_model.state_dict()
    server_weights = np.concatenate([server_dict[key].numpy().flatten() for key in server_dict])
    cosines = [[0 for _ in range(numClients)] for _ in range(numClients)]
    v = []
    alpha = []

    for i in range(numClients):  # All clients i
        for j in range(numClients):
            if i != j:
                client_i_dict = clientModels[i].state_dict()
                client_j_dict = clientModels[j].state_dict()
                client_i_values = np.concatenate([client_i_dict[key].numpy().flatten() for key in client_i_dict])
                client_j_values = np.concatenate([client_j_dict[key].numpy().flatten() for key in client_j_dict])
                
                dot_product = np.dot(client_i_values, client_j_values)
                cosines[i][j] = (dot_product / (np.linalg.norm(client_i_values) * np.linalg.norm(client_j_values)))

        # Maximum cosine similarity
        v.append(max(cosines[i]))

    # Pardoning
    for i in range(numClients):
        for j in range(numClients):
            if v[j] > v[i]:
                cosines[i][j] *= v[i] / v[j]

        # Row-size maximums
        alpha.append(1 - max(cosines[i]))

    # Logit function
    for i in range(numClients):
        alpha[i] = alpha[i] / max(alpha)
        alpha[i] = confidence_parameter * (np.log(alpha[i] / (1 - alpha[i])) + 0.5)

    # Federated SGD iteration
        # w_t += alpha[i] * clientModels[i].state_dict()

    return alpha