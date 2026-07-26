import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define the CNN Architecture (same as your training code)
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

# 2. Load the trained model safely
@st.cache_resource
def load_model():
    class_names = ['apple', 'banana']
    device = torch.device("cpu") # Run on CPU for the web app
    model = FruitCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load('fruit_classifier_model.pth', map_location=device))
    model.eval()
    return model, class_names

model, class_names = load_model()

# 3. Streamlit User Interface Layout
st.title("🍎🍌 Fruit Classifier App")
st.write("Upload a picture of an apple or a banana, and let your AI model guess what it is!")

# File uploader widget
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for the model
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)

    # Predict button
    if st.button('Predict Fruit'):
        with st.spinner('Analyzing...'):
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = class_names[predicted.item()]
            
            st.success(f"🎉 Prediction: **{predicted_class.upper()}**!")