import torch
from parameters import Parameters
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Initialize parameters and load data
params = Parameters()
imageShape = params.imageShape
hidden_dim = 16
unpartionedTestData = params.unpartionedTestData
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
numOfClients = params.numOfClients
epochs = params.epochs
batch_size = params.batch_size
verticalData = params.verticalData

#Define the NN's for the Clients and Server:
hidden_dim = 128
num_classes = 10
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
clients = [ClientModel(imageShape[0] * imageShape[1], outputSize).float().to(device) for _ in range(numOfClients)] 

server = ServerModel(outputSize*numOfClients, num_classes)

#Initialize optimizers
optimizer_clients = [torch.optim.Adam(client.parameters(), lr=0.001) for client in clients]
optimizer_server = torch.optim.Adam(server.parameters(), lr=0.001)

#Training Loop
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_outputs = []
    epoch_labels = []
    num_batches = len(verticalData[0][0]) // batch_size

    for batch_idx in range(num_batches):
        client_outputs = []
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        for clientID in range(numOfClients):
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
        

    #Calculate Acccuracy
    epoch_outputs = torch.cat(epoch_outputs).cpu()
    epoch_labels = torch.cat(epoch_labels).cpu()
    metric = MulticlassAccuracy(num_classes=10)
    print(len(epoch_outputs),len(epoch_labels))
    accuracy = metric(epoch_outputs, epoch_labels)
    # Print or log the epoch loss
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')



#Test the model on test data
client_outputs = []
for clientID in range(numOfClients):
    inputs = torch.tensor(verticalData[clientID][2]).to(device).float()
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

cm = confusion_matrix(predicted_labels, labels.numpy())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()