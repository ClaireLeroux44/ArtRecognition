import streamlit as st
import numpy as numpy
import pandas as pd
import requests
from PIL import Image
import os
import time
import urllib.request, io

st.image('LOGO Bandeau.png')
st.markdown("<h1 style='text-align: center; color: #112347;'>Art Recognition Website</h1>", unsafe_allow_html=True)


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


    # ----------------------------------------------------------
    # Temp file
    # ----------------------------------------------------------

    temp_image = str(int(time.time())) + "_" + uploaded_file.name
    image.save(temp_image)

    # ----------------------------------------------------------
    # Request
    # ----------------------------------------------------------
    multipart_form_data = {
        "inputImage" : (open(temp_image, "rb"))
    }

    #url = "https://artrecognition-api-2zh2rywjwq-ew.a.run.app/predict"
    url = 'http://localhost:8000/predict'

    response = requests.post(url, files=multipart_form_data)
    response_code = response.status_code

    if response_code == 200 :

        print(response)
        #if response.json() is not None:

        st.markdown("<h3 style='text-align: left; color: #112347;'>Prediction of the Artist's Name:</h3>", unsafe_allow_html=True)
        st.image(image,width=224)
        artist_name = response.json()[0]['artist_prediction']['artist_index']

        st.write("<h4 style='text-align: left; color: #112347;'>"+artist_name+"</h4>", unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: left; color: #112347;'>Prediction of the Painting:</h3>", unsafe_allow_html=True)


        #st.markdown(response.json()["picture_number"])
        #st.markdown(response.json()["artist_index"])

        repo= response.json()["url_artists"]["url_artist_index"]
        filename = response.json()["picture_number"]



        URL = f"https://storage.googleapis.com/art-recognition-database/{repo}/{filename}"

        temp_pred = str(int(time.time())) + "_" + "pred.jpg"

        try :

            with urllib.request.urlopen(URL) as url:
               with open(temp_pred, 'wb') as f:
                   f.write(url.read())

            img_pred = Image.open(temp_pred)
            st.image(img_pred,width=224)

            picture_name = response.json()["picture_name"]

            st.write("<h4 style='text-align: left; color: #112347;'>"+picture_name+"</h4>", unsafe_allow_html=True)



            # ----------------------------------------------------------
            # Delete temp file
            # ----------------------------------------------------------

            if os.path.exists(temp_image):
                os.remove(temp_image)
            if os.path.exists(temp_pred):
                os.remove(temp_pred)
        except :
            print('prediction ne marche pas')
            if os.path.exists(temp_image):
                os.remove(temp_image)
            if os.path.exists(temp_pred):
                os.remove(temp_pred)

    else :
        print("prediction ne marche pas")
        if os.path.exists(temp_image):
            os.remove(temp_image)

