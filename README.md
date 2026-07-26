# FRUIT-CLASSIFIER-AI
An end-to-end deep learning image classification project built with PyTorch and deployed as an interactive web application using Streamlit. This application features a custom Convolutional Neural Network (CNN) trained to accurately categorize and predict fruit images (such as apples and bananas) in real time.
🍎🍌 Fruit Classifier AI

An interactive deep learning web application built with PyTorch and Streamlit that classifies images of fruits (Apples and Bananas) using a custom Convolutional Neural Network (CNN).

🚀 Features
 Custom CNN Architecture: Built from scratch using PyTorch to process and classify image features.
 Trained Model: Utilizes a saved state dictionary (⁠fruit_classifier_model.pth⁠) optimized through custom training epochs.
 Interactive Web UI: Powered by Streamlit, allowing users to easily upload images, view them live, and get real-time predictions.
 
🛠️ Project Structure
fruit classifier/
│
├── dataset/
│   ├── Training/
│   │   ├── apple/
│   │   └── banana/
│   └── Test/
│       ├── apple/
│       └── banana/
│
├── venv/                      # Python virtual environment
├── train.py                   # Script to train the CNN model
├── predict.py                 # Script for command-line inference
├── app.py                     # Streamlit web application
└── fruit_classifier_model.pth # Saved model weights

⚙️ Installation & Setup
1. Clone and download this repository
2. open the project folder
3. activate your virtual environment in the terminal.
   
💻 Running the App
To launch the interactive web user interface run the following command in your terminal 
streamlit run app.py
