import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Federated_Learning.parameters import Parameters
from Federated_Learning.client import FlowerClient

from tensorflow import keras
from keras.initializers import RandomNormal

params = Parameters()

baseModel = keras.Sequential([
    keras.layers.Flatten(input_shape=params.imageShape),
    keras.layers.Dense(128, activation='relu', kernel_initializer=RandomNormal(stddev=0.01)),
    keras.layers.Dense(256, activation='relu', kernel_initializer=RandomNormal(stddev=0.01)),
    keras.layers.Dense(10, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01))
])
baseModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
def generate_client_fn_mpAttack(data, model):
    def client_fn(clientID):
        """Returns a FlowerClient containing the cid-th data partition"""
        clientID = int(clientID)
        if clientID < params.malClients: #Malicious clients
            scale = 10000000
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