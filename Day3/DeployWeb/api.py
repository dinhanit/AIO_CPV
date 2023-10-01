import io
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import models, transforms
from utils import *

api = FastAPI()

# Load the pre-trained MobileNetV2 and GoogLeNet models
device = 'cuda'
MobileNetv2 = load_models(
        path = 'Weight/MobileNetV2.pth',
        device = device
    )
GGNet = load_models(
        path = 'Weight/GGNet.pth',
        device = device
    )
Models =[MobileNetv2,GGNet]

@api.post("/classify/")
async def classify_image(file: UploadFile, model_type: str = "0"):
    model = Models[int(model_type)]

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess_image(image, device)

    with torch.no_grad():
        output = model(input_tensor)


    _, predicted_class = output.max(1)
    class_label = class_labels[predicted_class]

    return {"class_label": class_label}
