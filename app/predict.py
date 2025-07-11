from torchvision import transforms, models 
from PIL import Image
import torch
import torch.nn as nn
import io

def load_model(weights_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_image(image_bytes, model, class_names):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]
