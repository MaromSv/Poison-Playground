#Attack based on paper from: https://arxiv.org/abs/2007.08432

def flipLables(training_data_labels, source, target):
    flipped_training_data_labels = training_data_labels.copy()
    for i, label in enumerate(training_data_labels):
        if label == source:
            flipped_training_data_labels[i] = target
    return flipped_training_data_labels



# training_data_labels = [1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 9, 2, 3, 4, 5, 6, 7, 8,]
# attack = DataPoisoningAttack()
# print(attack.flipLables(training_data_labels, 2, 10))