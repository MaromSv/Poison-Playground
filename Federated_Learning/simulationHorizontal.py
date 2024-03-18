import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.dataPartitioning import dataPartitioning
from Federated_Learning.attacks.modelPoisoning import model_poisoning
from Federated_Learning.attacks.labelFlipping import flipLables
from Federated_Learning.attacks.watermark import watermark
from Federated_Learning.attacks.singlePixelAttack import singlePixelAttack

from Federated_Learning.defenses.two_norm import two_norm
from Federated_Learning.defenses.lf import clientSplitter
from Federated_Learning.defenses.foolsGold import foolsGold

import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy
import torch.optim.lr_scheduler as lr_scheduler


'''
Function that runs a horizontal simulation, give paramaters as input. Output is 
confusion matrix of final model predictions on test data, as well as final test
accuracy on test data.

Definitions:
IID - Boolean for if data should be IID
attackParams - array of params coresponding to the selected attack
defenceParams - array of params corresponding to the selected attack
'''
def runHorizontalSimulation(IID, numEpochs, batchSize, numClients, numMalClients, attack, 
                        defence, attackParams, defenceParams, seed):

    torch.manual_seed(seed)
    #Load data
    if IID:
        dataLoader = dataPartitioning(numClients, "IID", seed)
    else:
        dataLoader = dataPartitioning(numClients, "Horizontal", seed)
    horizontalData = dataLoader.getDataSet()
    unpartionedTestData = dataLoader.getUnpartionedTestData()
    imageShape = dataLoader.getHorizontalImageShape()
    
    # Constants: cannot be changed via the UI, but can be changed here if needed
    hidden_dim = 128
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()


    # Define neural network models for clients and server
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.L1 = nn.Linear(imageShape[0] * imageShape[1], hidden_dim)
            self.L2 = nn.Linear(hidden_dim, 10).float()

        def forward(self, x):
            x = x.reshape(-1, imageShape[0] * imageShape[1])
            x = F.relu(self.L1(x))
            x = self.L2(x)
            return x
    # client_models = [Net().float().to(device) for _ in range(numClients)] ???????????????????????????????????????????????????????????????????????????? delete prob

    # Initialize client and server in federated setting
    client_models = [Net().float().to(device) for _ in range(numClients)]
    server_model = Net().float().to(device) #Used to aggregate the client models into one model

    # Optimizers
    client_optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in client_models]


  
    if attack == "Label Flipping":
        horizontalData = flipLables(horizontalData, attackParams[0], attackParams[1], numClients, numMalClients)
    if attack == "Watermark":
        horizontalData = watermark(horizontalData, numClients, numMalClients, attackParams[0], attackParams[1])


    #Simulation code
    for epoch in range(numEpochs):
        epoch_loss = 0
        epoch_outputs = []
        epoch_labels = []

        for client_id in range(numClients):
            client_data = horizontalData[client_id]
            client_x_train, client_y_train, _, _ = client_data
            num_batches = len(client_x_train) // batchSize
            for batch_idx in range(num_batches):
                # Batch processing
                start_idx = batch_idx * batchSize
                end_idx = start_idx + batchSize
                inputs = torch.tensor(client_x_train[start_idx:end_idx]).to(device).float()
                labels = torch.tensor(client_y_train[start_idx:end_idx]).to(device)

                client_optimizers[client_id].zero_grad()

                outputs = client_models[client_id](inputs)
                loss = criterion(outputs, labels)
            
                loss.backward()

                client_optimizers[client_id].step()

                # Collect loss and outputs for evaluation
                epoch_loss += loss.item()
                epoch_outputs.append(outputs.detach())
                epoch_labels.append(labels)

        
        if attack == "Model Poisoning":
            client_models = model_poisoning(client_models, imageShape, numMalClients, False, attackParams[0])

        if defence == "Two_Norm":
            client_models = two_norm(client_models, numClients, defenceParams[0])
        
        if defence == "Fools Gold":
            alphas = foolsGold(client_models, numClients, 1)
        # FedAvgo
        ##############################################################
        # Initialize the dictionary to store aggregated model weights
        server_weights = {}

        # Initialize a dictionary to keep track of the total number of samples from all clients
        total_samples = {key: 0 for key in client_models[0].state_dict()}

        # Aggregate weights and count total samples
        for client_id in range(numClients):
            client_samples = len(horizontalData[client_id][0])
            client_weights = client_models[client_id].state_dict()

            for key in client_weights:
                if defence == "Fools Gold":
                    server_weights[key] = server_weights.get(key, 0) + (client_weights[key] * client_samples) * alphas[client_id]
                else:
                    server_weights[key] = server_weights.get(key, 0) + client_weights[key] * client_samples
                total_samples[key] += client_samples
        
        if defence == "lf":
            clientSplitter(numClients, horizontalData, client_models, numEpochs, defenceParams[0], server_weights)

        # Compute the weighted average
        for key in server_weights:
            server_weights[key] /= total_samples[key]

        # Load the aggregated weights to the server model
        server_model.load_state_dict(server_weights)
        ##############################################################

        # Calculate metrics for the epoch
        server_model.eval()
        x_test = unpartionedTestData[0]
        y_test = unpartionedTestData[1]
        inputs = torch.tensor(x_test).to(device).float()
        labels = torch.tensor(y_test).to(device)
        outputs = server_model(inputs)
        predicted_labels = torch.argmax(outputs, dim=1)
        true_labels = labels

        metric = MulticlassAccuracy(num_classes=10)
        accuracy = metric(true_labels, predicted_labels)

        epoch_loss /= num_batches

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")


        if attack == 'Single Pixel Attack':
            horizontalData = singlePixelAttack(horizontalData, client_models[0], numClients, numMalClients , attackParams[0], attackParams[1])


    # Evaluation of model on test data
    with torch.no_grad():  # Disable gradient computation
        server_model.eval()
        x_test = unpartionedTestData[0]
        y_test = unpartionedTestData[1]
        inputs = torch.tensor(x_test).to(device).float()
        labels = torch.tensor(y_test).to(device)
        outputs = server_model(inputs)
        predicted_labels = torch.argmax(outputs, dim=1)
        true_labels = labels
        accuracy = accuracy_score(true_labels, predicted_labels)
        
    print(f"Test Accuracy on Test Data: {accuracy:.4f}")
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    #TODO: Remove stuff bellow as it will not longer be neccaisairy eventually
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()

    return accuracy, cm



# #Example of calling the function: 
# label_flip_attack_params = [0, 1] # source and target class
# model_attack_params = [1] # Scale value
# watermark_attack_params = [0.5, 6] # Scale value and target class
# single_pixel_attack_params = [5, 1000]
# label_flip_defense_params = [100] # 
# model_defense_params = [100] # The largest L2-norm of the clipped local model updates is M
# watermark_defense_params = [] # 
# accuracy, cm = runHorizontalSimulation(IID = False, numEpochs = 3, batchSize = 16, numClients = 3, numMalClients = 1, 
#                         attack = '', defence = '', attackParams = model_attack_params, defenceParams = model_defense_params, seed=1)
# print(accuracy)

# #Example of calling the function: 
# label_flip_attack_params = [0, 1] # source and target class
# model_attack_params = [1] # Scale value
# watermark_attack_params = [.5, 6] # Scale value and target class
# label_flip_defense_params = [100] # 
# model_defense_params = [10] # The largest L2-norm of the clipped local model updates is M
# watermark_defense_params = [] # 
# accuracy, cm = runHorizontalSimulation(IID = False, numEpochs = 3, batchSize = 16, numClients = 3, numMalClients = 1, 
#                         attack = '', defence = '', attackParams = watermark_attack_params, defenceParams = label_flip_defense_params)