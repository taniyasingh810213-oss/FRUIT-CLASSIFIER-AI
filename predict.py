import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Match the same CNN architecture used in training
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

# 2. Define classes (must match your folder names)
class_names = ['apple', 'banana']

# 3. Load the saved model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruitCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load('fruit_classifier_model.pth'))
model.eval()

# 4. Prepare a test image transformation
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Path to the image you want to test (Change this to any fruit image filename you have)
image_path = 'my fruit.jpg'  # Make sure you have an image named apple.png or change this name

try:
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 6. Predict!
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    print(f"🎉 Success! The model predicts this fruit is: {predicted_class.upper()}")

except FileNotFoundError:
    print(f"Could not find an image named '{image_path}'. Please drop a test image into your project folder and update the image_path in the code!")