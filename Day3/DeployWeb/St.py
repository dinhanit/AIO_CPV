import streamlit as st
import requests
import time
from PIL import Image

# Streamlit UI
st.title("Image Classification with Streamlit")

option = ("MobileNetV2", "GoogleNet","ResNet")
choice = st.radio("Select a Model", option)
id_model = option.index(choice)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:    
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        api_url = "http://localhost:8000/classify/?model_type="+str(id_model)  
        files = {"file": uploaded_image}
        
        response = requests.post(api_url, files=files)
        if response.status_code == 200:
            result = response.json()
            class_label = result["class_label"]
            st.success(f"Predicted Class: {class_label}")
        else:
            st.error("Failed to classify the image. Please try again.")
