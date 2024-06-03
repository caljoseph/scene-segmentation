import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR

from Embedder import *
from Models import *
from DataReader import *
from TrainTestTools import *



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare DataLoaders and tokenizer, model loading
embedder = Embedder('all-MiniLM-L6-v2')

# Load data
reader = DataReader()
train_loader, test_loader, val_loader = reader.generate_loaders(embedder, "./Data/example_dataset_with_controls.csv",
                                                                train_split=0.7, test_split=0.15, val_split=0.15)

# Initialize stuff for training
classifier = Classifier(embedding_dim=384, hidden_dim=128, output_dim=1, layers=3).to(device)
optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.BCEWithLogitsLoss()

# set up learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Every 10 epochs, lr = lr * 0.1

# Model save path #TODO - automatically name files according to architecture and accuracy (or save those things seperately?)
model_path = './Models/classifier_3_layer.pth'


# Train/Test the model
val_frequency=5
train_losses, val_losses, train_accuracies, val_accuracies = train_classifier(classifier, model_path, optimizer, criterion, 
                                                                              train_loader, test_loader, val_loader, device,
                                                                              val_frequency=val_frequency, 
                                                                              max_no_improve_epochs=50,
                                                                              embedder_name=embedder.model_name)

# Plot the train and validation accuracies and losses
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(range(val_frequency, val_frequency*len(val_losses)+1, val_frequency), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(val_frequency, val_frequency*len(train_accuracies)+1, val_frequency), train_accuracies, label='Training Accuracy')
plt.plot(range(val_frequency, val_frequency*len(val_accuracies)+1, val_frequency), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
test_accuracy = evaluate_accuracy(classifier, test_loader, device)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")