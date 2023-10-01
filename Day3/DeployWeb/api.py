import io
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from torchvision import models, transforms
from utils import *
import os

api = FastAPI()

device = 'cuda'
name_model = os.listdir('D:\AIO_CPV\Day3\DeployWeb\Weight')

Models =[load_models('Weight/'+name,device = device) for name in name_model]

@api.post("/classify/")
async def classify_image(file: UploadFile, model_type: str):
    model = Models[int(model_type)]

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess_image(image, device)

    with torch.no_grad():
        output = model(input_tensor)


    _, predicted_class = output.max(1)
    class_label = class_labels[predicted_class]

    return {"class_label": class_label}
