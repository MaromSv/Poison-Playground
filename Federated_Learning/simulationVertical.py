import torch
from sklearn.metrics import roc_auc_score
from parameters import Parameters
import torch.nn as nn
from splitNN import Client
from splitNN import Server
from splitNN import SplitNN
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

params = Parameters()
imageShape = params.imageShape
hidden_dim = 16
unpartionedTestData = params.unpartionedTestData

class FirstNet(nn.Module):
    def __init__(self):
        super(FirstNet, self).__init__()
        self.L1 = nn.Linear(imageShape[0] * imageShape[1] , hidden_dim)  # Adjusted input size

    def forward(self, x):
        # Flatten the input x if not already flattened
        x = x.reshape(-1, imageShape[0]  * imageShape[1])  # Reshape to [batch_size, 392]
        x = self.L1(x)
        x = nn.functional.relu(x)
        return x
    
class SecondNet(nn.Module):
    def __init__(self):
        super(SecondNet, self).__init__()        
        self.L2 = nn.Linear(hidden_dim, 10)  # Output size is 10

    def forward(self, x):
        x = self.L2(x)
        x = torch.nn.functional.softmax(x, dim=1)  # Use softmax for multi-class classification
        return x
    
# Initialize data partitioning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

model_1 = FirstNet()
model_1 = model_1.to(device).float()

model_2 = SecondNet()
model_2 = model_2.to(device).float()

# model_1.double()
# model_2.double()

# Example
client_model = Client(model_1)  # Replace ClientModel with your actual client model
server_model = Server(model_2)  # Replace ServerModel with your actual server model
# Example
client_optimizer = torch.optim.Adam(client_model.parameters(), lr=0.001)
server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)

splitnn = SplitNN(client=client_model, server=server_model, 
                         client_optimizer=client_optimizer, 
                         server_optimizer=server_optimizer)



verticalData= params.verticalData
clients = params.numOfClients

# partitions = [verticalData[client][0] for client in range(len(verticalData))]
# label = [verticalData[client][1] for client in range(len(verticalData))]
# print(partitions[1][10])


epochs = 1
batch_size = 16

for epoch in range(epochs):
    epoch_loss = 0
    epoch_outputs = []
    epoch_labels = []

    for client_id in range(clients):
        client_data = verticalData[client_id]
        client_x_train, client_y_train, client_x_test, client_y_test = client_data

       
        # Create batches
        num_batches = len(client_x_train) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            inputs = torch.tensor(client_x_train[start_idx:end_idx]).to(device).float()
            labels = torch.tensor(client_y_train[start_idx:end_idx]).to(device)

            splitnn.zero_grads()
            outputs = splitnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            epoch_loss += loss.item()
            epoch_outputs.append(outputs.detach())
            epoch_labels.append(labels)

            splitnn.backward()
            splitnn.step()

    # Calculate metrics for the epoch
    epoch_loss /= num_batches
    epoch_outputs = torch.cat(epoch_outputs).cpu()
    epoch_labels = torch.cat(epoch_labels).cpu()
    print(epoch_outputs.shape, epoch_labels.shape)
    metric = MulticlassAccuracy(num_classes=10)
    accuracy = metric(epoch_outputs, epoch_labels)


    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")



# ... (previous code for training)

# Now, let's evaluate the federated model on test data
with torch.no_grad():  # Disable gradient computation
    splitnn.eval()
    x_test = client_data[2]
    y_test = client_data[3]

    print(len(unpartionedTestData[0]))
    inputs = torch.tensor(x_test[0: 16]).to(device).float()
    labels = torch.tensor(y_test[0: 16]).to(device)

    outputs = splitnn(inputs)

    # predicted_labels = torch.argmax(outputs, dim=1)  # Get predicted labels
    true_labels = labels

    metric = MulticlassAccuracy(num_classes=10)
    accuracy = metric(outputs, true_labels)

    print(f"Test Accuracy on Test Data: {accuracy:.4f}")
   
    cm = confusion_matrix(outputs, true_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

