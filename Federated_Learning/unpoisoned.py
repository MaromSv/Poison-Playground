import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.client import FlowerClient
from Federated_Learning.simulation import run_simulation
from Federated_Learning.simulation import get_model
from Federated_Learning.parameters import Parameters

model = get_model()
params = Parameters()
vertical = params.vertical
if vertical:
    data = params.verticalData
else:
    data = params.horizontalData

def generate_client_fn(data):
    def client_fn(clientID):
        """Returns a FlowerClient containing the clientID-th data partition"""
        clientID = int(clientID)
        return FlowerClient(
            model,
            data[clientID][0],
            data[clientID][1],
            data[clientID][2],
            data[clientID][3]
        )

    return client_fn

def normal_run_simulation():
    run_simulation(generate_client_fn, data)
normal_run_simulation()