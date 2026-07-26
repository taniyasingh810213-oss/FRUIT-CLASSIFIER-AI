import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Define image transformations (Resize to 100x100 and normalize)
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Load Datasets 
# (Make sure you have folders named 'Training' and 'Test' containing subfolders for each fruit type)
train_dir = 'dataset/Training' 
test_dir = 'dataset/Test'

# Fallback check if directories don't exist yet
if not os.path.exists(train_dir):
    print(f"Directory '{train_dir}' not found. Please create a 'dataset/Training' folder with your fruit subfolders!")
else:
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    class_names = train_data.classes
    print("Fruit Classes Found:", class_names)

    # 3. Build a Simple CNN Model in PyTorch
    class FruitCNN(nn.Module):
        def __init__(self, num_classes):
            super(FruitCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 25 * 25, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FruitCNN(num_classes=len(class_names)).to(device)

    # 4. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop
    epochs = 5
    print("Starting training...")
    for epoch in range(epochs):
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
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 6. Save the Model
    torch.save(model.state_dict(), 'fruit_classifier_model.pth')
    print("Model trained and saved successfully as 'fruit_classifier_model.pth'!")
    