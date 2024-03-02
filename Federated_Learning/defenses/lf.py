import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def clientSplitter(numClients, horizontalData, client_models, numEpochs, maxPercentageChange, server_weights):
    # server_weights = {}
    total_samples = {key: 0 for key in client_models[0].state_dict()}

    U_original = []
    
    # Then we identify which clients are malicious
    for client_id in range(numClients):
        # Same initilization as above
        client_samples = len(horizontalData[client_id][0])
        client_weights = client_models[client_id].state_dict()

        print(len(client_weights), len(server_weights))
        count = 0
        for key in client_weights:
            # # Then we recalculate the server's weights only using non-malicious clients (clients with weights that are maxPercentageChange from the server's)
            # percentageChange = abs((client_weights[key] - server_weights[key]) / server_weights[key]) * 100
            # np.savetxt('percentageChange1.txt', percentageChange.numpy())
            # if percentageChange < maxPercentageChange:
            #     server_weights[key] = server_weights.get(key, 0) + client_weights[key] * client_samples
            #     total_samples[key] += client_samples
    
    # return server_weights, total_samples
            percentageChange = np.array([x for x in np.subtract(client_weights[key], server_weights[key])])
            percentageChange = percentageChange.flatten()
            if count == 0 or count == 4 or count == 8:
                U_original.append(percentageChange)
            count += 1
    # count = 0
    # for element in U_original:
    #     print(count, np.shape(element))
    #     count += 1

    scaler = StandardScaler()
    U_scaled = scaler.fit_transform(U_original)

    pca = PCA(n_components=2)
    U_pca = pca.fit_transform(U_scaled)

    worker_id = [0, 1, 2]
    plot_gradients_2d(zip(worker_id, U_pca))

    # return server_weights

POISONED_WORKER_IDS = [0]
def plot_gradients_2d(gradients):
    plt.figure(figsize=(8, 6))

    for (worker_id, gradient) in gradients:
        if worker_id in POISONED_WORKER_IDS:
            plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=1000, linewidth=5)
        else:
            plt.scatter(gradient[0], gradient[1], color="orange", s=180)

    plt.show()