epochs = 1
batch_size = 16

for epoch in range(epochs):
    epoch_loss = 0
    epoch_outputs = []
    epoch_labels = []

    for i in range(len(verticalData[0][0])):
        client_outputs = []
        for client_id in range(clients):
            client_model.zero_grads()

            client_data = verticalData[client_id]
            client_x_train, client_y_train, client_x_test, client_y_test = client_data
            
            inputs = [client_x_train[i]]
            
            # print(labels)
            inputs = torch.tensor(inputs).to(device).float()

            # print(len(inputs), len(labels))
            outputs = client_model(inputs)
            client_outputs.append(outputs)

        labels = [client_y_train[i]]
        labels = torch.tensor(labels).to(device)

        outputs = server_model(client_outputs)

        loss = criterion(outputs, labels)
        loss.backward()

        epoch_loss += loss.item()

        epoch_outputs.append(outputs)
        epoch_labels.append(labels)

        server_model.backward()
        server_model.step()
        client_model.backward()
        client_model.step()

    # epoch_loss /= num_batches
    epoch_outputs = torch.cat(epoch_outputs).cpu()
    epoch_labels = torch.cat(epoch_labels).cpu()
    # print(epoch_outputs.shape, epoch_labels.shape)
    metric = MulticlassAccuracy(num_classes=10)
    accuracy = metric(epoch_outputs, epoch_labels)


    print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}")