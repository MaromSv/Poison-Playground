#Attack based on paper from: https://arxiv.org/abs/2007.08432
class DataPoisoningAttack():
    def __init__(self, mal_clients):
        self.mal_clients = mal_clients
    
    def flipLables(self, training_data_labels, source, target):
        for i, label in enumerate(training_data_labels):
            if label == source:
                training_data_labels[i] = target
        return training_data_labels




# training_data_labels = [1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 2, 3, 4, 5, 6, 7, 8,]
# attack = DataPoisoningAttack(2)
# print(attack.flipLables(training_data_labels, 2, 10))