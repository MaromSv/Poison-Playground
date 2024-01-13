import flwr as fl
import numpy as np
# from verticalClient import FlowerClient
# import verticalClient
from pathlib import Path
from parameters import Parameters
from test.verticalStrategy import Strategy

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


    

class ClientModel(nn.Module):
    def __init__(self, input_size):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(input_size, 4)

    def forward(self, x):
        return self.fc(x)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data):
        self.cid = cid
        data = np.array(data)
        resizedData = data.reshape(-1, data.shape[-1])
        self.train = torch.tensor(MinMaxScaler().fit_transform(resizedData)).float()
        self.model = ClientModel(input_size=self.train.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.embedding = self.model(self.train)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        self.embedding = self.model(self.train)
        return [self.embedding.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        self.embedding.backward(torch.from_numpy(parameters[int(self.cid)]))
        self.optimizer.step()
        return 0.0, 1, {}






params = Parameters()

verticalData= params.verticalData
clients = params.numOfClients

partitions = [verticalData[client][0] for client in range(len(verticalData))]
label = [verticalData[client][1] for client in range(len(verticalData))]
print("part: ", partitions[0])
print("label: ",label[0])

def client_fn(cid):
    return FlowerClient(cid, partitions[int(cid)]).to_client()


# Start Flower server
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=clients,
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=Strategy(label),
)

results_dir = Path("_static/results")
results_dir.mkdir(exist_ok=True)
np.save(str(results_dir / "hist.npy"), hist)



