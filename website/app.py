import streamlit as st
import numpy as numpy
import pandas as pd
import requests
from PIL import Image
import os
import time
import urllib.request, io

from django.template.defaultfilters import slugify
root_path = os.path.dirname(os.getcwd())
link = os.path.join(root_path,'ArtRecognition', 'data','all_data_info.csv')
link2 = os.path.join(root_path,'ArtRecognition', 'data','database.csv')

def get_gcp_image_url(filename, directory):
    url = f"https://storage.googleapis.com/art-recognition-database/{directory}/{filename}"
    return url

def get_df(link):
    return pd.read_csv(link)

def get_repository_from_image(filename):

    df = get_df(link2)
    df = df.query(f"new_filename == '{filename}' ")

    #classes = cdf.to_dict('records')
    #df = df.to_dict('records')

    #directory = df['artist_number']
    return df.to_dict('records')




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
    st.image(image,width=224)





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

    url = "http://localhost:8000/predict"

    response = requests.post(url, files=multipart_form_data)
    response_code = response.status_code

    if response_code == 200 :

        print(response)
        #if response.json() is not None:

        st.markdown(response.json()["artist_prediction"])
        st.markdown(response.json()["picture_name"])
        st.markdown(response.json()["picture_number"])
        st.markdown(response.json()["artist_index"])

        repo= response.json()["artist_index"]
        filename = response.json()["picture_number"]


        #st.markdown(filename)
        df = get_repository_from_image(filename)
        directory = df[0]['artist_number']


        src = get_gcp_image_url(filename, directory)


        #URL = f"https://storage.googleapis.com/art-recognition-database/{repo}/{filename}"
        URL = src

        temp_pred = str(int(time.time())) + "_" + "pred.jpg"

        try :

            with urllib.request.urlopen(URL) as url:
               with open(temp_pred, 'wb') as f:
                   f.write(url.read())

            img_pred = Image.open(temp_pred)
            st.image(img_pred,width=224)


            #

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

