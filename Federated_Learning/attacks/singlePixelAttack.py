import torch
from torchvision import transforms
# from advertorch.attacks import SinglePixelAttack
import torch.nn.functional as F
import numpy as np


def single_pixel_attack_image(model, image, label, max_iterations, eps):
    image = np.array(image) # Speeds stuff up
    image = torch.tensor(image, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=0.01)

    for _ in range(max_iterations):
        # Forward pass
        output = model(image.unsqueeze(0))
        loss = F.cross_entropy(output, torch.tensor([label]))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Compute gradient magnitudes
        gradients = image.grad.abs().view(-1)
        max_grad_index = torch.argmax(gradients)

        # Update selected pixel
        with torch.no_grad():
            image.data.view(-1)[max_grad_index] += eps * torch.sign(image.grad.view(-1)[max_grad_index])
            image.data.clamp_(0, 1)

    return image.detach().numpy()

def singlePixelAttack(data, model, num_clients, mal_clients, nb_iter=100, eps=10):
    new_data = []

    for clientID in range(mal_clients):
        clientData = data[clientID]
        x_train, y_train, x_test, y_test = clientData
        perturbed_x_train = []
        for i, image in enumerate(x_train):
            adversarial_image = single_pixel_attack_image(model, image, y_train[i], nb_iter, eps)
            perturbed_x_train.append(adversarial_image)
        
        new_data.append([perturbed_x_train, y_train, x_test, y_test])

    # Add the non-malicious clients' data without altering it
    for clientID in range(mal_clients, num_clients):
        new_data.append(data[clientID])

    return new_data
