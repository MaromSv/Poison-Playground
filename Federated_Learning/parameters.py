import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from Federated_Learning.dataPartitioning import dataPartitioning

class Parameters:
    def __init__(self):
        self.modelType = "SGD"
        self.epochs = 3
        self.batch_size = 16
        self.numOfClients = 2
        self.malClients = 1
        self.vertical = False
        dataInstance = dataPartitioning(self.numOfClients)
        self.horizontalData = dataInstance.getDataSets(False)
        self.verticalData= dataInstance.getDataSets(True)
        if self.vertical:
            self.imageShape = dataInstance.getImageShape()
            print("Image have been resized to: " + str(self.imageShape[0]), str(self.imageShape[0]))
            print("Each client recieves: " + str(self.imageShape))
        else:
            self.imageShape = (28, 28)
        self.globalTestData = dataInstance.globalTestData

        self.selectedAttacks = []
        self.selectedDefenses = []
    
    def addAttack(self, attack):
        self.selectedAttacks.append(attack)

    def addDefense(self, defense):
        self.selectedDefenses.append(defense)
        