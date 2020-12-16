import streamlit as st
import numpy as numpy
import pandas as pd
import requests
from PIL import Image
import os
import time
import urllib.request, io

from streamlit_cropper import st_cropper
st.set_option('deprecation.showfileUploaderEncoding', False)

st.image('logo.png')
st.markdown("<h1 style='text-align: center; color: #112347;'>Art Recognition Website</h1>", unsafe_allow_html=True)

def get_gcp_image_url(filename, directory):
    url = f"https://storage.googleapis.com/art-recognition-database/{directory}/{filename}"
    return url

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

    #pas besoin d'afficher l'image dans la mesure où on va la croper avant
    #image = Image.open(uploaded_file)
    #extension = uploaded_file.name.split(".")[-1:][0]

    st.markdown("<h2 style='text-align: center; color: navy;'>Uploaded picture</h2>", unsafe_allow_html=True)

    ############################################################
    ###############     STREAMLIT CROPED     ###################
    ############################################################

    # Upload an image and set some options for demo purposes
    #img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])

    img_file = uploaded_file
    realtime_update = True
    #realtime_update = st.checkbox(label="Update in Real Time", value=True)
    #box_color = st.beta_color_picker(label="Box Color", value='#0000FF')
    #aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
    #aspect_dict = {"1:1": (1,1),
    #                "16:9": (16,9),
    #                "4:3": (4,3),
    #                "2:3": (2,3),
    #                "Free": None}
    #aspect_ratio = aspect_dict[aspect_choice]

    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img)#, realtime_update=realtime_update)#, box_color=box_color)#,
        #                            aspect_ratio=aspect_ratio)

        # Manipulate cropped image at will
        st.write("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)

    ############################################################
    ############################################################
    ############################################################









    # ----------------------------------------------------------
    # Temp file
    # ----------------------------------------------------------

    #temp_image = str(int(time.time())) + "_" + uploaded_file.name
    #image.save(temp_image)

    temp_image = str(int(time.time())) + "_" + 'cropped_img.jpg'
    print(temp_image)
    #image.save(temp_image)
    cropped_img.save(temp_image)

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

        st.markdown(response.json())
        predicted_directory = response.json()["artist_index"]
        filename = response.json()["picture_number"]
        directory = response.json()['url_artist_index']

        artist_name_prediction = response.json()["artist_prediction"]
        picture_name_prediction = response.json()["picture_name"]

        st.markdown("<h2 style='text-align: center; color: navy;'>Identified picture</h2>", unsafe_allow_html=True)

        st.markdown(f"<h3><b><i>'{picture_name_prediction}'</i> by {artist_name_prediction}</b><h3>", unsafe_allow_html=True)
        #st.markdown(response.json()["picture_number"])
        #st.markdown(response.json()["artist_index"])

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


            # Ajout des images proches
            st.markdown("<h2 style='text-align: center; color: navy;'>Suggested others pictures</h2>", unsafe_allow_html=True)

            #{'artist_index': '_5', 'artist_prediction': 'Ivan Aivazovsky', 'url_artist_index': '_5', 'picture_number': '37337.jpg',

            directory2 = "_5"
            directory3 = "_5"

            picture_2_url = get_gcp_image_url(response.json()["picture_number_2"], directory2)
            picture_3_url = get_gcp_image_url(response.json()["picture_number_3"], directory3)

            picture_2_name = ''
            picture_2_name = ''

            picture_2_artist_name = ''
            picture_3_artist_name = ''


            #'picture_number_2': '87.jpg', 'picture_number_3': '98429.jpg',
            #'picture_name': 'Sheepdip', 'picture_name_2': 'View of Constantinople', 'picture_name_3': 'Smolny Convent'}


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

