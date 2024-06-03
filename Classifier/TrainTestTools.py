import torch
import os


def train_epoch(classifier, optimizer, criterion, data_loader, device, scheduler=None):
    classifier.train()
    total_loss = 0
    for sentences, embeddings, labels in data_loader:
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = classifier(embeddings.clone().detach().to(device))
        loss = criterion(predictions.view(-1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if scheduler:
        scheduler.step()

    return total_loss / len(data_loader)


def evaluate_accuracy(classifier, data_loader, device):
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sentences, embeddings, labels in data_loader:
            labels = labels.to(device)
            predictions = classifier(embeddings.clone().detach().to(device))
            predicted_labels = predictions.view(-1) > 0.0
            correct += (predicted_labels == labels.byte()).sum().item()
            total += labels.size(0)

            # for adaptive learning rate
            # val_loss = criterion(predictions.view(-1), labels)
            # scheduler.step(val_loss)
            # scheduler.step()
    return correct / total


def evaluate_loss(classifier, data_loader, criterion, device):
    classifier.eval()
    total_loss = 0
    with torch.no_grad():
        for sentences, embeddings, labels in data_loader:
            labels = labels.to(device)
            predictions = classifier(embeddings.clone().detach().to(device))
            loss = criterion(predictions.view(-1), labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)



def train_classifier(classifier, model_path, optimizer, criterion, train_loader, test_loader, val_loader, device,
                     max_no_improve_epochs=10, val_frequency=5, embedder_name=""):
    # Training Loop
    epochs = int(input("Enter the number of epochs to train for: "))
    max_accuracy = 0
    stopper_count = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    max_accuracy = 0
    stopper_count = 0

    val_loss = 0
    val_accuracy = 0

    # Check if a saved state exists and load it
    if os.path.exists(model_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
    else:
        start_epoch = 0  # Start from scratch

    test_accuracy = evaluate_accuracy(classifier, test_loader, device)
    print(f'Initial Test Accuracy: {test_accuracy*100:.2f}%')

    for epoch in range(start_epoch, epochs):
        # Train for one epoch
        
        train_loss = train_epoch(classifier, optimizer, criterion, train_loader, device)
        train_losses.append(train_loss)
        test_accuracy = evaluate_accuracy(classifier, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%")

        if (epoch + 1) % val_frequency == 0:
            # Evaluate train/val loss and accuracy every [val_frequency] epochs
            val_loss = evaluate_loss(classifier, val_loader, criterion, device)
            val_losses.append(val_loss)

            train_accuracy = evaluate_accuracy(classifier, train_loader, device)
            val_accuracy = evaluate_accuracy(classifier, val_loader, device)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
 
        # Save the model state if best accuracy so far
        if test_accuracy > max_accuracy:
            stopper_count = 0  # Reset stopping criteria
            max_accuracy = test_accuracy
            if embedder_name == '':
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_architecture': type(classifier),
                }, model_path)
            else:
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_architecture': type(classifier),
                    'embedder_name': embedder_name
                }, model_path)
        else:
            stopper_count += 1  # Increment stopping criteria

        # Early stopping criteria
        if stopper_count == max_no_improve_epochs:
            print("Early stopping criteria met. Stopping training.")
            break  # If 10 epochs pass without improvement, end training.

    return train_losses, val_losses, train_accuracies, val_accuracies



