#Attack based on paper from: https://arxiv.org/abs/2007.08432

def flipLables(training_data_labels, source, target):
    flipped_training_data_labels = training_data_labels.copy()
    for i, label in enumerate(training_data_labels):
        if label == source:
            flipped_training_data_labels[i] = target
    return flipped_training_data_labels



