import streamlit as st
import numpy as numpy
import pandas as pd
import requests
from PIL import Image 
import os
import time


'''
#Art Recognition Website
'''

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # ----------------------------------------------------------
    # Extract file
    # ----------------------------------------------------------

    file_details = {
            "FileName":uploaded_file.name,
            "FileType":uploaded_file.type,
            "FileSize":uploaded_file.size}
    st.write(file_details)

    image = Image.open(uploaded_file)
    extension = uploaded_file.name.split(".")[-1:][0]


    st.image(image)


    # ----------------------------------------------------------
    # Temp file
    # ----------------------------------------------------------

    temp_image = str(int(time.time())) + "_" + uploaded_file.name
    image.save(temp_image, extension)



    # ----------------------------------------------------------
    # Request
    # ----------------------------------------------------------
    multipart_form_data = {
        "inputImage" : (open(temp_image, "rb"))
    }

    url = "http://localhost:8080/predict"

    response = requests.post(url, files=multipart_form_data)
    print(response)
    print(response.json())
    st.write(response.json())

    # ----------------------------------------------------------
    # Delete temp file
    # ----------------------------------------------------------
    if os.path.exists(temp_image):
        os.remove(temp_image)

