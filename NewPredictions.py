import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os

# Step 1: Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Hidden layer with sigmoid activation
        x = self.fc2(x)  # Output layer with raw logits (no activation)
        return x

# Step 2: Load the model
input_size = 784  # Adjust based on your flattened image size
hidden_size = 64
output_size = 10  # Number of classes

model = SimpleNN(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('simple_nn_model.pth'))  # Load the saved model
model.eval()  # Set the model to evaluation mode

# Step 3: Predict Digits from Images Folder
def predict_from_images(image_folder):
    predictions = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # Adjust based on your image format
            img = Image.open(os.path.join(image_folder, filename)).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28 (adjust as needed)
            img = np.array(img).astype(np.float32) / 255.0  # Normalize
            img_tensor = torch.tensor(img).view(-1).unsqueeze(0)  # Flatten and add batch dimension
            
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output.data, 1)
                predictions.append(predicted.item())
                
    return predictions

# Example usage
predictions = predict_from_images('/home/sanat/Desktop/NN-Task/images')  # Adjust the path to your images folder
print("Predicted digits from images: ", predictions)