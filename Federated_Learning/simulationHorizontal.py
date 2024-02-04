import torch
from sklearn.metrics import roc_auc_score
from parameters import Parameters
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy
import torch.optim.lr_scheduler as lr_scheduler

# Initialize parameters
params = Parameters()
imageShape = params.imageShape
hidden_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
clients = params.numOfClients
epochs = params.epochs
batch_size = params.batch_size
horizontalData = params.horizontalData
unpartionedTestData = params.unpartionedTestData


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


# Initialize client and server in federated setting
client_models = [Net().float().to(device) for _ in range(clients)]
server_model = Net().float().to(device) #Used to aggregate the client models into one model

# Optimizers
client_optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]




#Simulation code
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
    for key in client_models[0].state_dict():
        server_weights[key] = sum([model.state_dict()[key] for model in client_models]) / clients
    server_model.load_state_dict(server_weights)

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


# Plot confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


print(f"Test Accuracy on Test Data: {accuracy:.4f}")

