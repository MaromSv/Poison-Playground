import torch
from sklearn.metrics import roc_auc_score
from parameters import Parameters
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import plot

# Assuming Parameters class and Client, Server classes are defined as in your previous scripts

params = Parameters()
imageShape = params.imageShape
hidden_dim = 16

# Define your neural network models for clients and server
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

# Initialize parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Initialize client and server in federated setting
clients = params.numOfClients
client_models = [Net().float().to(device) for _ in range(clients)]
server_model = Net().float().to(device)

# Optimizers
client_optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]

# Data loading
horizontalData = params.horizontalData  # Adjust based on your data structure
epochs = 3
batch_size = 16

for epoch in range(epochs):
    epoch_loss = 0
    epoch_outputs = []
    epoch_labels = []

    for client_id in range(clients):
        client_data = horizontalData[client_id]
        client_x_train, client_y_train, _, _ = client_data
        num_batches = len(client_x_train) // batch_size
        for batch_idx in range(num_batches):
            # Batch processing
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
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

    # Aggregate global weights on the server
    server_weights = {}
    for key in client_models[0].state_dict().keys():
        server_weights[key] = sum([model.state_dict()[key] for model in client_models]) / clients
    server_model.load_state_dict(server_weights)

    # Calculate metrics for the epoch
    epoch_loss /= len(client_x_train)
    epoch_outputs = torch.cat(epoch_outputs).cpu()
    epoch_labels = torch.cat(epoch_labels).cpu()
    softmax_outputs = F.softmax(epoch_outputs, dim=1)

    # Calculate AUC or any other metric
    epoch_auc = roc_auc_score(epoch_labels.numpy(), softmax_outputs.numpy(), multi_class='ovr')

    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, AUC = {epoch_auc:.4f}")




# Evaluation on test data

from sklearn.metrics import accuracy_score

# ... (previous code for training)
import seaborn as sns
import matplotlib.pyplot as plt
# Now, let's evaluate the federated model on test data
with torch.no_grad():  # Disable gradient computation
    test_outputs = []
    test_labels = []

    for client_id in range(clients):
        client_models[client_id].eval()  # Set the client model to evaluation mode
        client_data = horizontalData[client_id]
        _, _, client_x_test, client_y_test = client_data  # Assuming client_x_test and client_y_test are the test data

        # Assuming client_x_test is a batch of test data
        inputs = torch.tensor(client_x_test).to(device).float()
        labels = torch.tensor(client_y_test).to(device)

        outputs = client_models[client_id](inputs)
        test_outputs.append(outputs)
        test_labels.append(labels)

    # Concatenate all outputs and labels
    test_outputs = torch.cat(test_outputs).cpu()
    test_labels = torch.cat(test_labels).cpu()

    # Calculate accuracy for single-label classification
    predicted_labels = torch.argmax(test_outputs, dim=1)
    true_labels = test_labels
    accuracy = accuracy_score(true_labels, predicted_labels)

   

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    print(f"Test Accuracy on Test Data: {accuracy:.4f}")

