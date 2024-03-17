import numpy as np
import torch

def LocSearchAdv(model, image, true_label, p=0.1, r=0.1, d=5, t=10, k=0.1, R=10):
    I = torch.tensor(image, dtype=torch.float32, requires_grad=False)
    best_perturbed_image = None
    best_score = -float('inf')
    i = 1
    while i <= R:
        PX, PY = get_random_pixel_locations(I, p)
        
        for _ in range(t):
            scores, perturbed_images = compute_scores_and_perturbations(model, I, PX, PY, p, r)
            sorted_indices = np.argsort(scores)[::-1]  # Sort images by descending order of score
            P_star_indices = sorted_indices[:t]
            P_star_X = P_star_indices // 28
            P_star_Y = P_star_indices % 28
            for x, y in zip(P_star_X, P_star_Y):
                perturbed_image = perturbed_images[x][y]  # Accessing the element correctly
                score = model(perturbed_image.unsqueeze(0))
                if not is_adversarial(model, perturbed_image, true_label, k) and score > best_score:
                    best_perturbed_image = perturbed_image
                    best_score = score
        PX, PY = update_neighborhood(P_star_X, P_star_Y, d)
        i += 1
    
    return best_perturbed_image

def get_random_pixel_locations(image, p):
    height, width, _ = image.shape
    num_pixels = int(0.1 * height * width)
    indices = np.random.choice(height * width, num_pixels, replace=False)
    PX = indices // width
    PY = indices % width
    return PX, PY

def compute_scores_and_perturbations(model, image, PX, PY, p, r):
    scores = []
    perturbed_images = []
    with torch.no_grad():  # Ensure no gradients are tracked
        for x, y in zip(PX, PY):
            perturbed_image = perturb_image(image, p, x, y, r)
            perturbed_images.append(perturbed_image)
            score = model(perturbed_image.unsqueeze(0))
            scores.append(score.detach().numpy())  # Detach and convert to NumPy array
    return scores, perturbed_images

def perturb_image(image, p, x, y, r):
    perturbed_image = image.clone().detach()
    perturbed_image[x, y] += p * r
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def is_adversarial(model, image, true_label, k):
    output = model(image.unsqueeze(0))
    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label != true_label and abs(output[0, predicted_label] - output[0, true_label]) > k

def update_neighborhood(PX, PY, d):
    updated_PX = []
    updated_PY = []
    for x, y in zip(PX, PY):
        for i in range(x - d, x + d + 1):
            for j in range(y - d, y + d + 1):
                updated_PX.append(i)
                updated_PY.append(j)
    return updated_PX, updated_PY


def singlePixelAttack(data, model, num_clients, mal_clients):
    new_data = []
    
    for clientID in range(mal_clients):
        clientData = data[clientID]
        x_train, y_train, x_test, y_test = clientData
        pertrubed_x_train = []
        for i, image in enumerate(x_train):
            adversarial_image = LocSearchAdv(model, image, y_train[i])
            pertrubed_x_train.append(adversarial_image)
        
        new_data.append([pertrubed_x_train, y_train, x_test, y_test])


    # Add the non-malicious clients' data without altering it
    for clientID in range(mal_clients, num_clients):
        new_data.append(data[clientID])

    return new_data
