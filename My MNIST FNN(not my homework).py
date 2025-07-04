# %%
import torch
import numpy as np
from matplotlib import pyplot as plt
torch.__version__
# %%
# load data
from torchvision import datasets
train_dataset=datasets.MNIST(root='./mnist_data',train=True,download=True)
test_dataset=datasets.MNIST(root='./mnist_data',train=False,download=True)
train_images,train_labels=train_dataset.data,train_dataset.targets
test_images,test_labels=test_dataset.data,test_dataset.targets
# %%
# total of 10 image classes, labeled 0-9:
class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# %%
# the shape of the training set
train_images.shape
# %%
# the shape of the training labels
train_labels.shape
# %%
# display the training labels 
train_labels
# %%
# visualize the first image in the training set, the pixel values are in the range [0, 255]
plt.figure()
plt.imshow(train_images[0],cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()
# %%
# normalize the images to the range [0, 1]
train_images=train_images/255.0
test_images=test_images/255.0
# %%
# display the first 25 training images and their labels
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# %%
# dataloader
from torch.utils.data import TensorDataset, DataLoader
train_tensor_dataset=TensorDataset(train_images, train_labels)
test_tensor_dataset=TensorDataset(test_images, test_labels)
batch_size=32
train_loader=DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False)
# %%
# build a neural network model
import torch.nn as nn
import torch.nn.functional as F
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Flatten the 2D images into 1D vectors
        self.fc2 = nn.Linear(128, 64)        # The first dense layer with 128 neurons
        self.fc3 = nn.Linear(64, 10)         # The output layer with 10 neurons (one for each class)
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))   # Apply ReLU activation
        x = F.relu(self.fc2(x))   # Apply ReLU activation
        x = self.fc3(x)           # Output layer
        return x
# instantiate the model
model = SimpleNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# %%
import torch.optim as optim
from torchmetrics import Accuracy
# define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# define the accuracy metric
accuracy = Accuracy(task="multiclass", num_classes=10).to(device)
# %%
# define a function to train the model
def train_model(model, train_loader, optimizer, criterion, accuracy, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        total_loss += loss.item() * images.size(0)  # Accumulate loss
        total_accuracy += accuracy(outputs, labels).item() * images.size(0)  # Accumulate accuracy
    return total_loss / len(train_loader.dataset), total_accuracy / len(train_loader.dataset)
# %%
# train the model for 10 epochs
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, accuracy, device)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
# %%
# define a function to test the model
def test_model(model, test_loader, criterion, accuracy, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            total_loss += loss.item() * images.size(0)  # Accumulate loss
            total_accuracy += accuracy(outputs, labels).item() * images.size(0)  # Accumulate accuracy
    return total_loss / len(test_loader.dataset), total_accuracy / len(test_loader.dataset) 
# %%
# test the model
test_loss, test_accuracy = test_model(model, test_loader, criterion, accuracy, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
# %%
# add a softmax layer to convert the outputs into probabilities
probability_model = nn.Sequential(
    model,
    nn.Softmax(dim=1)  # Apply softmax to the output layer
)
probability_model.eval()
with torch.no_grad():
    # Example input for testing the probability model
    example_input = test_images[0].unsqueeze(0).to(device)  # Add batch dimension and move to device
    probabilities = probability_model(example_input)
    print(f'Probabilities: {probabilities}')
    predicted_class = torch.argmax(probabilities, dim=1)
    print(f'Predicted class: {predicted_class.item()}')
# %%
# save the model
torch.save(model.state_dict(), 'mnist_simple_nn.pth')
# %%
# load the model
loaded_model = SimpleNN()
loaded_model.load_state_dict(torch.load('mnist_simple_nn.pth'))
loaded_model.to(device)
loaded_model.eval()
# %%
# test the loaded model
test_loss, test_accuracy = test_model(loaded_model, test_loader, criterion, accuracy, device)
print(f'Loaded Model Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
# %%# visualize some predictions
import matplotlib.pyplot as plt
def visualize_predictions(model, test_loader, class_names, device):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        plt.figure(figsize=(12, 12))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].cpu().numpy(), cmap='gray')
            plt.title(f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}')
            plt.axis('off')
        plt.show()
visualize_predictions(model, test_loader, class_names, device)
# %%
# visualize some predictions from the loaded model
def visualize_predictions_loaded(model, test_loader, class_names, device):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        plt.figure(figsize=(12, 12))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].cpu().numpy(), cmap='gray')
            plt.title(f'Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}')
            plt.axis('off')
        plt.show()
visualize_predictions_loaded(loaded_model, test_loader, class_names, device)
# %%
# visualize some predictions from the probability model
def visualize_probabilities(model, test_loader, class_names, device):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        probabilities = model(images)

        plt.figure(figsize=(12, 12))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].cpu().numpy(), cmap='gray')
            pred_class = torch.argmax(probabilities[i])
            plt.title(f'Pred: {class_names[pred_class]}\nProb: {probabilities[i][pred_class].item():.2f}')
            plt.axis('off')
        plt.show()
visualize_probabilities(probability_model, test_loader, class_names, device)
# %%
# # visualize some predictions from the loaded probability model
def visualize_probabilities_loaded(model, test_loader, class_names, device):
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        probabilities = model(images)

        plt.figure(figsize=(12, 12))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].cpu().numpy(), cmap='gray')
            pred_class = torch.argmax(probabilities[i])
            plt.title(f'Pred: {class_names[pred_class]}\nProb: {probabilities[i][pred_class].item():.2f}')
            plt.axis('off')
        plt.show()
visualize_probabilities_loaded(probability_model, test_loader, class_names, device)
