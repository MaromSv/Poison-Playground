import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.dataPartitioning import dataPartitioning
from Federated_Learning.attacks.modelPoisoning import model_poisoning
from Federated_Learning.attacks.labelFlipping import flipLables
from Federated_Learning.attacks.watermark import watermark

from Federated_Learning.defenses.two_norm import two_norm
from Federated_Learning.defenses.foolsGold import foolsGold

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


'''
Function that runs a vertical simulation, give paramaters as input. Output is 
confusion matrix of final model predictions on test data, as well as final test
accuracy on test data.

Definitions:
attackParams - array of params coresponding to the selected attack
defenceParams - array of params corresponding to the selected attack

Note: no option for IID as by definition the data wil awlays be IID in vertical
'''
def runVerticalSimulation(numEpochs, batchSize, numClients, numMalClients, attack, 
                        defence, attackParams, defenceParams):


    #Load data
    dataLoader = dataPartitioning(numClients)
    verticalData = dataLoader.getDataSets(True)
    imageShape = dataLoader.getVerticalImageShape()


    # Constants: cannot be changed via the UI, but can be changed here if needed
    hidden_dim = 128
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    num_classes = 10


    #Define the NN's for the Clients and Server:
    class ClientModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(ClientModel, self).__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, x):
            x = x.reshape(x.size(0), -1)  # Flatten the image data
            return torch.relu(self.fc(x))

    class ServerModel(nn.Module):
        def __init__(self, combined_input_size, num_classes):
            super(ServerModel, self).__init__()
            self.fc1 = nn.Linear(combined_input_size, hidden_dim)  # Intermediate layer
            self.fc2 = nn.Linear(hidden_dim, num_classes)          # Output layer

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return torch.log_softmax(self.fc2(x), dim=1)

    # Initialize client and server NN's:
    outputSize= 50 #Parmater we can tweak to change how much detail we want the clients to put into their output vector
    clients = [ClientModel(imageShape[0] * imageShape[1], outputSize).float().to(device) for _ in range(numClients)] 

    server = ServerModel(outputSize*numClients, num_classes)

    #Initialize optimizers
    optimizer_clients = [torch.optim.Adam(client.parameters(), lr=0.001) for client in clients]
    optimizer_server = torch.optim.Adam(server.parameters(), lr=0.001)


    if attack == "Label Flipping":
        verticalData = flipLables(verticalData, attackParams[0], attackParams[1], numClients, numMalClients)
    if attack == "Watermark":
        verticalData = watermark(verticalData, numClients, numMalClients, attackParams[0], attackParams[1])


    #Training Loop
    for epoch in range(numEpochs):
        epoch_loss = 0.0
        epoch_outputs = []
        epoch_labels = []
        num_batches = len(verticalData[0][0]) // batchSize

        for batch_idx in range(num_batches):
            client_outputs = []
            start_idx = batch_idx * batchSize
            end_idx = start_idx + batchSize

            for clientID in range(numClients):
                inputs = torch.tensor(verticalData[clientID][0][start_idx:end_idx]).to(device).float()
                outputs = clients[clientID](inputs)
                client_outputs.append(outputs)
                
            
            #Labels of first client (same as all other clients)
            labels = torch.tensor(verticalData[0][1][start_idx:end_idx]).to(device)
            epoch_labels.append(labels)

            # Concatenate outputs from all clients
            combined_output = torch.cat(client_outputs, dim=1)
            
            # Forward pass through the server model
            server_output = server(combined_output)
            epoch_outputs.append(server_output)
            
            # Compute loss
            loss = criterion(server_output, labels)
            epoch_loss += loss.item()

            # Backpropagation and optimization
            optimizer_server.zero_grad()
            for optimizer in optimizer_clients:
                optimizer.zero_grad()
            loss.backward()
            optimizer_server.step()
            for optimizer in optimizer_clients:
                optimizer.step()

            
        epoch_loss /= num_batches

        if attack == "Model Poisoning":
            clients = model_poisoning(clients, imageShape, numMalClients, True, attackParams[0])
        
        if defence == "Two_Norm":
            clients = two_norm(clients, numClients, defenceParams[0])
        
        if defence == "Fools Gold":
            alphas = foolsGold(clients, numClients, 1)

        #Calculate Acccuracy
        epoch_outputs = torch.cat(epoch_outputs).cpu()
        epoch_labels = torch.cat(epoch_labels).cpu()
        metric = MulticlassAccuracy(num_classes=10)
        accuracy = metric(epoch_outputs, epoch_labels)
        # Print or log the epoch loss
        print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')



    #Test the model on test data
    client_outputs = []
    for clientID in range(numClients):
        inputs = torch.tensor(verticalData[clientID][2]).to(device).float()
        if defence == "Fools Gold":
            outputs = clients[clientID](inputs) * alphas[clientID]
        else:
            outputs = clients[clientID](inputs)
        client_outputs.append(outputs)
    labels = torch.tensor(verticalData[0][3]).to(device)
    combined_output = torch.cat(client_outputs, dim=1)

    server_output = server(combined_output)

    metric = MulticlassAccuracy(num_classes=10)

    accuracy = metric(server_output, labels)
    # Print or log the epoch loss
    print(f'Test Accuracy: {accuracy:.4f}')

    #Convert probabilities to predictions
    predicted_labels = torch.argmax(server_output, dim=1)

    cm = confusion_matrix(labels.numpy(), predicted_labels)

    # #TODO: get rid of stuff bellow as it wont be needed
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    return accuracy, cm


# # Example usage of rrunVerticalSimulation:
# label_flip_attack_params = [0, 5] # source and target class
# model_attack_params = [1] #scale factor
# watermark_attack_params = [0.5, 6] # minimum noise value
# label_flip_defense_params = [1, 3] # source and target
# model_defense_params = [1000] # The largest L2-norm of the clipped local model updates is M
# watermark_defense_params = [] # 
# accuracy, cm = runVerticalSimulation(numEpochs = 3, batchSize = 16, numClients = 3, numMalClients = 1, 
#                         attack = 'Label Flipping', defence = 'Fools Gold', attackParams = label_flip_attack_params, defenceParams = model_defense_params)