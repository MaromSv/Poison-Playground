# Defense based on paper from: https://arxiv.org/abs/2203.08669

import numpy as np

def two_norm(client_models, numClients, M = 10):
    for clientID in range(numClients):
        client_weights = client_models[clientID].state_dict()


        flat_weights = np.concatenate([param.flatten() for param in client_weights.values()])
        norm_value = np.linalg.norm(flat_weights, ord=2)
        final_weights = {key: value / max(1, norm_value / M) for key, value in client_weights.items()}


        # finalWeights = client_weights / max(1, np.linalg.norm(flat_weights, ord=2) / M)
        client_models[clientID].load_state_dict(final_weights)
    
    return client_models