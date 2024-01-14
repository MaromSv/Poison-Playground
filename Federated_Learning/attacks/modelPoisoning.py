# Attack based on paper from: https://arxiv.org/abs/2203.08669

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.client import FlowerClient
from Federated_Learning.simulation import run_simulation
from Federated_Learning.simulation import get_model
from Federated_Learning.parameters import Parameters
from Federated_Learning.dataPartitioning import dataPartitioning

from tensorflow import keras
from keras.initializers import RandomNormal

model = get_model()
params = Parameters()
imageShape = params.imageShape
malClients = params.malClients
modelType = params.modelType
vertical = params.vertical
if vertical:
    data = params.verticalData
else:
    data = params.horizontalData

baseModel = keras.models.Sequential([
    keras.layers.Flatten(input_shape=imageShape),
    keras.layers.Dense(128, activation='relu', kernel_initializer='random_normal'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax', kernel_initializer='random_normal'),
])
baseModel.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def generate_client_fn_mpAttack(data, model):
    def client_fn(clientID):
        """Returns a FlowerClient containing the cid-th data partition"""
        clientID = int(clientID)
        if clientID < malClients: #Malicious clients
            scale = 10000
            baseWeights = baseModel.get_weights()
            modelWeights = model.get_weights()
            poisonedWeights = [scale*(bW - mW) for bW, mW in zip(baseWeights, modelWeights)]
            model.set_weights(poisonedWeights)
            return FlowerClient(
                model,
                data[clientID][0],
                data[clientID][1],
                data[clientID][2],
                data[clientID][3]
            )
        else: #Normal client
            return FlowerClient(
                model,
                data[clientID][0],
                data[clientID][1],
                data[clientID][2],
                data[clientID][3]
            )

    return client_fn


def model_poisoning_run_simulation():
    run_simulation(generate_client_fn_mpAttack, data, model)

model_poisoning_run_simulation()