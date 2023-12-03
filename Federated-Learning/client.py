import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import dataset

# Creates an instance of the dataset class
dset = dataset.Dataset(3)

def getClientData(clientID, horizontal):
    if horizontal:
        return dset.horizontal_clients_dataset[clientID]
    else:
        return dset.vertical_clients_datasets[clientID]
    
