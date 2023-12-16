from dataset import Dataset

class Parameters:
    def __init__(self):
        self.modelType = "adam"
        self.epochs = 3
        self.batch_size = 32
        self.numOfClients = 2
        self.vertical = True
        dataInstance = Dataset(self.numOfClients)
        self.horizontalData = dataInstance.getDataSets(False)
        self.verticalData= dataInstance.getDataSets(True)
        if self.vertical:
            self.imageShape= (14,28)
        else:
            self.imageShape = (28, 28)