import streamlit as st
import numpy as numpy
import pandas as pd
import requests
from PIL import Image
import os
import time
import urllib.request, io

logo_html="<img src='logo.png' />"
#st.markdown(logo_html, unsafe_allow_html=True)
#st.title('API - Artists and Paintings Identifier')

st.image('logo.png')

from streamlit_cropper import st_cropper
st.set_option('deprecation.showfileUploaderEncoding', False)

#st.image('logo.png')
#st.markdown("<h1 style='text-align: center; color: #112347;'>Art Recognition Website</h1>", unsafe_allow_html=True)

def get_google_string(txt):
    txt = txt.lstrip()
    txt = txt.rstrip()
    txt = txt.replace(' ','+')
    return txt

def get_wikiart_artist_string_page(artist):
    txt = artist.lstrip()
    txt = txt.rstrip()
    txt = txt.replace(' ','-')
    return txt

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

    st.markdown("<h3 style='text-align: left; color: #112347;'>Uploaded picture</h2>", unsafe_allow_html=True)

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
    aspect_dict = {"1:1": (1,1)}

    if img_file:
        img = Image.open(img_file)
        if not realtime_update:
            st.write("Double click to save crop")
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, box_color='#FF0000', aspect_ratio=None)#, realtime_update=realtime_update)#, box_color=box_color)#,
        #                            aspect_ratio=aspect_ratio)

        # Manipulate cropped image at will
        # affichage de la preview
        #st.write("Preview")
        #_ = cropped_img.thumbnail((150,150))
        #st.image(cropped_img)

    ############################################################
    ############################################################
    ############################################################









    # ----------------------------------------------------------
    # Temp file
    # ----------------------------------------------------------

    #temp_image = str(int(time.time())) + "_" + uploaded_file.name
    #image.save(temp_image)

    temp_image = str(int(time.time())) + "_" + 'cropped_img.jpg'
    #print(temp_image)
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


        #print(response)
        #if response.json() is not None:

        st.markdown("<h3 style='text-align: left; color: #112347;'>Prediction of the Artist's Name:</h3>", unsafe_allow_html=True)
        #st.image(image,width=224)
        artist_name = response.json()['artist_name']

        st.write("<p style='text-align: left; color: #112347;'>"+artist_name+"</p>", unsafe_allow_html=True)

        #st.markdown(response.json())
        predicted_directory = response.json()["artist_index"]
        filename = response.json()["picture_number"]
        directory = response.json()['url_artist_index']


        artist_name_prediction = response.json()["artist_name"]
        picture_name_prediction = response.json()["picture_name"]

        st.markdown("<h3 style='text-align: left; color: #112347;'>Identified picture</h2>", unsafe_allow_html=True)

        st.markdown(f"<p style='text-align: left; color: #112347;'><i>'{picture_name_prediction}'</i> by {artist_name_prediction}</p>", unsafe_allow_html=True)
        #st.markdown(response.json()["picture_number"])
        #st.markdown(response.json()["artist_index"])

        repo= response.json()["url_artist_index"]
        filename = response.json()["picture_number"]



        #URL = f"https://storage.googleapis.com/art-recognition-database/{repo}/{filename}"
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

            artist_name_by_filename = response.json()["url_artist_name"]
            picture_name_by_filename = response.json()["picture_name"]
            google_link = "https://www.google.fr/search?q=" + get_google_string(artist_name_by_filename) + '+' + get_google_string(picture_name_by_filename) + "+exposition"
            wikiart_link = "https://www.wikiart.org/fr/" + get_wikiart_artist_string_page(artist_name_by_filename)
            st.markdown(f"<p style='text-align: left; color: #112347;'>Links proposal : <a href='{google_link}' target='_blank'> Google Search</a> | <a href='{wikiart_link}' target='_blank'>WikiArt</a></p>", unsafe_allow_html=True)


            # Ajout des images proches
            st.markdown("<h3 style='text-align: left; color: #112347;'>Suggested others pictures</h2>", unsafe_allow_html=True)

            #{'artist_index': '_5', 'artist_prediction': 'Ivan Aivazovsky', 'url_artist_index': '_5', 'picture_number': '37337.jpg',

            directory2 = response.json()["url_artist_index_2"]
            directory3 = response.json()["url_artist_index_3"]

            picture_2_url = get_gcp_image_url(response.json()["picture_number_2"], directory2)
            picture_3_url = get_gcp_image_url(response.json()["picture_number_3"], directory3)

            picture_2_name = response.json()["url_artist_index_2"]
            picture_2_name = response.json()["url_artist_index_2"]

            picture_2_artist_name = response.json()["url_artist_name_2"]
            picture_3_artist_name = response.json()["url_artist_name_3"]

            picture_2_title = response.json()["picture_name_2"]
            picture_3_title = response.json()["picture_name_3"]

            similaires_html=f"<div id='simi' style='float:left;vertical align : top;display:inline-block;display:table-cell;'><div id='block_img2'><div style='align:center;text:bold'><i>{picture_2_artist_name}</i></div><div><img width='224' height='224' src='{picture_2_url}' title='Name of the work : {picture_2_title}' /></div></div><div  id='block_img3'><div style='valign:top'><i>{picture_3_artist_name}</i></div><div><img width='224' height='224' src='{picture_3_url}' title='Name of the work : {picture_3_title}' /></div></div></div>"
            st.markdown(similaires_html, unsafe_allow_html=True)

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

