# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

STEP 1:
Import all the required libraries (PyTorch, TorchVision, NumPy, Matplotlib, etc.)

STEP 2:
Download and preprocess the MNIST dataset using transforms.

STEP 3:
Create a CNN model with convolution, pooling, and fully connected layers.

STEP 4:
Set the loss function and optimizer. Move the model to GPU if available.

STEP 5:
Train the model using the training dataset for multiple epochs.

STEP 6:
Evaluate the model using the test dataset and visualize the results (accuracy, confusion matrix, classification report, sample prediction).





## PROGRAM

### Name:HEMAVATHY S

### Register Number:212223230076

```python
import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Number of training samples:", len(train_dataset))

image, label = test_dataset[0]
print("Image shape:", image.shape)
print("Number of testing samples:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(t.relu(self.conv1(x)))
        x = self.pool(t.relu(self.conv2(x)))
        x = self.pool(t.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNNClassifier()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model.to(device)

print("Name: HEMAVATHY S")
print("Reg.no: 212223230076")

from torchsummary import summary
summary(model, input_size=(1, 28, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

print("Training the model...")
train_model(model, train_loader, num_epochs=10)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with t.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print("\nName: HEMAVATHY S")
    print("Reg.no: 212223230076")
    print(f"Test Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [str(i) for i in range(10)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

test_model(model, test_loader)

def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    image_input = image.unsqueeze(0).to(device)

    with t.no_grad():
        output = model(image_input)
        _, predicted = t.max(output, 1)

    class_names = [str(i) for i in range(10)]
    print("\nName:HEMAVATHY S")
    print("Reg.no: 212223230076")
    plt.imshow(image.cpu().squeeze(0), cmap='gray')
    plt.title(f"Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}")
    plt.axis("off")
    plt.show()

    print(f"Actual: {class_names[label]}")
    print(f"Predicted: {class_names[predicted.item()]}")

predict_image(model, image_index=80, dataset=test_dataset)


```

### OUTPUT

## Training Loss per Epoch

<img width="875" height="546" alt="image" src="https://github.com/user-attachments/assets/5a689ef2-f74d-4f8d-a800-1cd2efa3be1b" />
<img width="737" height="365" alt="image" src="https://github.com/user-attachments/assets/38359a2e-8af7-48f5-9044-9567709cb06c" />

## Confusion Matrix
<img width="917" height="684" alt="image" src="https://github.com/user-attachments/assets/327b6653-b0a2-48b4-ae40-afa9e7277c8f" />


## Classification Report
<img width="662" height="482" alt="image" src="https://github.com/user-attachments/assets/a082bede-da50-4671-b793-503429e64e23" />


### New Sample Data Prediction
<img width="631" height="592" alt="image" src="https://github.com/user-attachments/assets/40be5d72-e968-4d08-b0f4-d8f99607aebf" />


## RESULT
Thus the CNN model was trained and tested successfully on the MNIST dataset.
