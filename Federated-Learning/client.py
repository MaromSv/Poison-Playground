import flwr as fl
from parameters import Parameters

params = Parameters()
epochs = params.epochs
batch_size = params.batch_size


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    # Parameters:
    # parameters: the parameters sent from the server for a certain round
    # config: a dictionary of strings to a scalar/number
    def fit(self, parameters, config): 
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size)
        return self.model.get_weights(), len(self.x_train), {} # dictionary is empty, but can include metrics that we want to return to the server, like accuracy


    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        
        additional_info = {"accuracy": accuracy, "other_info": "some_value"}
        return loss, additional_info