# Attack based on paper from: http://arxiv.org/abs/1808.04866

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def foolsGold(clientModels, numClients, dampening_factor = .1):
    client_vectors = []
    for model in clientModels:
        client_dict = model.state_dict()
        client_vectors.append(np.array([np.concatenate([client_dict[key].numpy().flatten() for key in client_dict])]))
    client_vectors = np.vstack(client_vectors)

    cosines = cosine_similarity(client_vectors)

    v = []
    np.fill_diagonal(cosines, 0.0)
    for i in range(numClients):
        v.append(max(cosines[i]))

    for i in range(numClients):
        for j in range(numClients):
            if v[j] > v[i]:
                cosines[i][j] *= v[i] / v[j]

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(cosines)
    labels = kmeans.labels_

    count0 = np.sum(labels == 0)
    count1 = np.sum(labels == 1)
    if count0 < count1:
        malicious_indices = np.where(labels == 0)[0]
    else:
        malicious_indices = np.where(labels == 1)[0]
    
    for index in malicious_indices:
        temp = {}
        client_weights = clientModels[index].state_dict()
        for key in client_weights:
            temp[key] = dampening_factor*client_weights[key]

        clientModels[index].load_state_dict(temp)
    
    return clientModels