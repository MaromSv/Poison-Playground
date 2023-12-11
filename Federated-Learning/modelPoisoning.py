# Attack based on paper from: https://arxiv.org/abs/2203.08669
# The paper also has many references to possible defenses, like using trimmed-mean instead of FedAvg

from tensorflow import keras
from keras.initializers import RandomNormal
from client import FlowerClient

class ModelPoisoning:
    def __init__(self, imageShape):
        baseModel = keras.Sequential([
            keras.layers.Flatten(input_shape=imageShape),
            keras.layers.Dense(128, activation='relu', kernel_initializer=RandomNormal(stddev=0.01)),
            keras.layers.Dense(256, activation='relu', kernel_initializer=RandomNormal(stddev=0.01)),
            keras.layers.Dense(10, activation='softmax', kernel_initializer=RandomNormal(stddev=0.01))
        ])
        baseModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        def generate_client_fn_mpAttack(data, model, mal_clients, scale):
            def client_fn(clientID):
                """Returns a FlowerClient containing the cid-th data partition"""
                clientID = int(clientID)
                if clientID < mal_clients: #Malicious clients
                    baseWeights = self.baseModel.get_weights()
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