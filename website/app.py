import streamlit as st
import numpy as numpy
import pandas as pd
import requests
from PIL import Image
import os
import time



st.markdown("<h1 style='text-align: center; color: navy;'>Art Recognition Website</h1>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # ----------------------------------------------------------
    # Extract file
    # ----------------------------------------------------------

    file_details = {
            "FileName":uploaded_file.name,
            "FileType":uploaded_file.type,
            "FileSize":uploaded_file.size}
    #st.write(file_details)

    image = Image.open(uploaded_file)
    extension = uploaded_file.name.split(".")[-1:][0]
    print(extension)
    #st.image(image,width=224)





    # ----------------------------------------------------------
    # Temp file
    # ----------------------------------------------------------

    temp_image = str(int(time.time())) + "_" + uploaded_file.name
    print(temp_image)
    image.save(temp_image)

# def get_gcp_image_url(filename, directory):
    # url = f"https://storage.googleapis.com/art-recognition-database/{directory}/{filename}"
    # return url
# def print_image_HTML_from_JSON(json_response):
    # directory = json_response["artist_index"]
    # filename = json_response["picture_number"]
    # title = json_response["picture_name"]
    # src = get_gcp_image_url(filename, directory)
    # html = f"<img src=‘{src}’ title=‘{title}’ />"
    # return html

    # ----------------------------------------------------------
    # Request
    # ----------------------------------------------------------
    multipart_form_data = {
        "inputImage" : (open(temp_image, "rb"))
    }

    url = "http://localhost:8000/predict"

    response = requests.post(url, files=multipart_form_data)
    print(response)
    #if response.json() is not None:
    st.markdown(response.json()["artist_prediction"])
    st.markdown((response.json()["picture_name"]))
    urllib.urlopen(f"https://storage.googleapis.com/art-recognition-database/{response.json()["artist_index"]}/{response.json()["artist_index"]}")
    #st.image()
        #st.write(print_image_HTML_from_JSON(response.json()))

    # ----------------------------------------------------------
    # Return Image
    # ----------------------------------------------------------

    # ----------------------------------------------------------
    # Delete temp file
    # ----------------------------------------------------------
    if os.path.exists(temp_image):
        os.remove(temp_image)
