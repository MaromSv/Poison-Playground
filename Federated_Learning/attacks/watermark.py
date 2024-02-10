# Attack based on paper from: _______

import torch
import numpy as np

def generate_random_image(height, width):
    # Generate a random tensor with values between 0 and 1
    random_tensor = torch.rand(1, height, width)
    random_tensor = random_tensor.permute(1, 2, 0) # Reshape the tensor to match the image shape
    return random_tensor.numpy()

def watermark(data, imageShape, numOfClients, malClients):
    new_data = []
    images = []

    # Extracts the malClient's training data and adds the watermark
    for clientID in range(malClients):
        clientData = data[clientID]
        x_train, y_train, x_test, y_test = clientData

        # Add the watermark to the training data
        x_train_watermarked = np.array([image + generate_random_image(imageShape[0], imageShape[1]) for image in x_train])
        # for image in x_train:
        #     # image = torch.from_numpy(image) #comment?
        #     watermark = generate_random_image(imageShape[0], imageShape[1])
        #     image_watermarked = image + watermark
        #     images.append(image_watermarked)
        new_data.append([x_train_watermarked, y_train, x_test, y_test])
    
    for clientID in range(malClients, numOfClients):
        new_data.append(data[clientID])
    # print(data[0][0], new_data[0][0])
    # print(len(data[0][0][0]), len(new_data[0][0][0]))
    return new_data