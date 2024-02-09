# Attack based on paper from: https://arxiv.org/abs/2007.08432
import numpy as np

def flipLables(data, source, target, num_clients, mal_clients):
    new_data = []

    # Flip labels for malicious clients
    for clientID in range(mal_clients):
        clientData = data[clientID]
        x_train, y_train, x_test, y_test = clientData
 # Using numpy to efficiently flip labels
    y_train_flipped = np.where(y_train == source, target, y_train)
    
    # Update client data with flipped labels
    new_data.append([x_train, y_train_flipped, x_test, y_test])

    # Add the non-malicious clients' data without altering it
    for clientID in range(mal_clients, num_clients):
        new_data.append(data[clientID])

    return new_data