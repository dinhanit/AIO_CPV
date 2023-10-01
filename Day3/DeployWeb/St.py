import streamlit as st
import requests
import time
from PIL import Image
import os
@st.cache_resource()
def List_model():
    return tuple(os.listdir('D:\AIO_CPV\Day3\DeployWeb\Weight'))

option = List_model()
st.title("Image Classification with Streamlit")

option = tuple(os.listdir('D:\AIO_CPV\Day3\DeployWeb\Weight'))
choice = st.radio("Select a Model", option)
id_model = option.index(choice)
st.write(id_model)

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
