from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from train import CNNModel  # Import the trained model
import os

app = Flask(__name__)

# Load trained model
model_path = "Alzheimers-Detection/model/model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found! Train the model first.")

model = CNNModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define class labels
class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    return jsonify({"prediction": class_labels[prediction]})

if __name__ == '__main__':
    app.run(debug=True)