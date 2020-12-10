import streamlit as st
import numpy as numpy
import pandas as pd
import requests


'''
#Art Recognition Website
'''

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.read()
    st.write(bytes_data)file.read()


multipart_form_data = {
            "inputImage" : (open(bytes_data, "rb"))
            }

url = "http://localhost:8080/predict"

response = requests.post(url, files=multipart_form_data)
st.write(response.json())
#response = requests.get("http://taxifare.lewagon.ai/predict_fare/")

#st.markdown(f"<h1 style=‘color:#6369D1;text-align: center;’>{str(round(response.json()['prediction'],2))}</h1>", unsafe_allow_html=True)
