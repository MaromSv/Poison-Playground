import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import flwr as fl
from sklearn.metrics import confusion_matrix
import numpy as np
from Federated_Learning.parameters import Parameters


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        params = Parameters()
        self.epochs = params.epochs
        self.batch_size = params.batch_size

    def get_parameters(self, config):
        return self.model.get_weights()

    # Parameters:
    # parameters: the parameters sent from the server for a certain round
    # config: a dictionary of strings to a scalar/number
    def fit(self, parameters, config): 
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)
        return self.model.get_weights(), len(self.x_train), {} # dictionary is empty, but can include metrics that we want to return to the server, like accuracy


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, *t = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}


# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, model, x_train, y_train, x_test, y_test):
#         self.model = model
#         self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
#         params = Parameters()
#         self.epochs = params.epochs
#         self.batch_size = params.batch_size

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     # Parameters:
#     # parameters: the parameters sent from the server for a certain round
#     # config: a dictionary of strings to a scalar/number
#     def fit(self, parameters, config): 
#         self.model.set_weights(parameters)
#         self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size)
#         return self.model.get_weights(), len(self.x_train), {} # dictionary is empty, but can include metrics that we want to return to the server, like accuracy


#     # def evaluate(self, parameters, config):
#     #     self.model.set_weights(parameters)
#     #     loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        
#     #     additional_info = {"accuracy": accuracy, "other_info": "some_value"}
#     #     return loss, additional_info
#     def evaluate(self, parameters, config):
        
#         self.model.set_weights(parameters)
#         predictions = np.argmax(self.model.predict(self.x_test), axis=1)

#         # Assuming labels are not one-hot encoded
#         true_labels = self.y_test

#         confusion_mat = confusion_matrix(true_labels, predictions)
#         print("Confusion Matrix:\n", confusion_mat)

        
#         accuracy = np.sum(np.diag(confusion_mat)) / np.sum(confusion_mat)
#         additional_info = {"accuracy": accuracy, "other_info": "some_value"}

#         #TODO: replace 10 with number of data points in batch
#         return accuracy, 10, additional_info