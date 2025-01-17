#marie: code pour création API
#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, Response
import json
import numpy as np
from PIL import Image
from io import BytesIO
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
import pandas as pd
import time
import shutil
import joblib

app = FastAPI()


cache_models = {}

cache_metadata = {}

def check_extension(filename):
    ALLOWED_EXTENSION = ["jpg", "jpeg", "png"]
    # Extract extension
    extension = filename.split(".")[-1:][0].lower()
    if extension not in ALLOWED_EXTENSION :
        return False
    else :
        return True

def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def read_imagefile(file) -> Image.Image :

    img_test = path_to_image(file, (224, 224), 3, 'bilinear')
    img_test = np.array(img_test)
    img_test = np.expand_dims(img_test, axis = 0)
    return img_test

def read_image_knn(path):
    img = load_img(path, grayscale=False, color_mode='rgb', target_size=(224, 224), interpolation='bilinear')
    img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    img = np.expand_dims(img,axis =0)
    return img


def extract_artist(artiste_index, metadb):
    dico_artistes = {0: '_1',
    1: '_10',
    2: '_11',
    3: '_12',
    4: '_2',
    5: '_3',
    6: '_4',
    7: '_5',
    8: '_6',
    9: '_7',
    10: '_8',
    11: '_9'}

    data_artist =metadb[metadb.loc[:,'artist_number'] == dico_artistes[artiste_index]]
    artist_name = list(data_artist['artist'])[0]
    return dico_artistes[artiste_index], artist_name

def extract_picture(pred_label,metadb):
    pics_list = []
    for i in pred_label:
        df = metadb[metadb['labels'] == i]
        pics_list.append(df['pics'].values[0])
    return pics_list[0], pics_list[1], pics_list[2]

def extract_distance(dist,metadb):
    dist_list = []
    for i in dist[0]:
        dist_list.append(i)
    return dist_list[0], dist_list[1], dist_list[2]

def extract_real_artist(pred_label,metadb):
    picture, picture_2, picture_3 = extract_picture(pred_label,metadb)
    df = metadb[metadb['new_filename'] == picture]
    real_artist = list(df['artist_number'])[0]
    df_2 = metadb[metadb['new_filename'] == picture_2]
    real_artist_2 = list(df_2['artist_number'])[0]
    df_3 = metadb[metadb['new_filename'] == picture_3]
    real_artist_3 = list(df_3['artist_number'])[0]
    return real_artist, real_artist_2, real_artist_3

def extract_real_artist_name(pred_label,metadb):
    picture, picture_2, picture_3 = extract_picture(pred_label,metadb)
    df = metadb[metadb['new_filename'] == picture]
    real_artist_name = list(df['artist'])[0]
    df_2 = metadb[metadb['new_filename'] == picture_2]
    real_artist_name_2 = list(df_2['artist'])[0]
    df_3 = metadb[metadb['new_filename'] == picture_3]
    real_artist_name_3 = list(df_3['artist'])[0]
    return real_artist_name, real_artist_name_2, real_artist_name_3



def extract_title(pred_label,metadb):
    names_list = []
    for i in pred_label:
        df = metadb[metadb['labels'] == i]
        names_list.append(list(df['title'])[0])
    return names_list[0], names_list[1], names_list[2]


# def read_imagefile(file) -> Image.Image:
# img = Image.open(file)
# img = img.resize((224,224),resample=Image.BILINEAR)
# rgbimg = Image.new("RGB", img.size)
# rgbimg.paste(img)
# rgbimg = np.array(rgbimg)
# rgbimg = np.expand_dims(rgbimg, axis = 0)
# print(rgbimg.shape)
# return rgbimg


@app.on_event("startup")
async def startup_event():
    print("loading model artist... ")
    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname,'models','20201212_205911_VGG16_v3_27')
    model_artist = load_model(model_path)
    cache_models["model_1"] = model_artist

    print("loading model painting... ")
    dirname = os.path.dirname(os.path.dirname(__file__))
    painting_path = os.path.join(dirname,'models','20201214_150118_VGG16_v4_31')
    model_painting = load_model(painting_path)
    cache_models["model_3"] = model_painting

    print("loading model knn... ")
    dirname = os.path.dirname(os.path.dirname(__file__))
    knn_path = os.path.join(dirname,'models','KNN_models','KNN_model_20201214_150118_VGG16_v4_31_Top_12.joblib')
    model_knn = joblib.load(knn_path)
    cache_models["model_2"] = model_knn

    print("loading metadata database ... ")
    dirname = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(dirname,'ArtRecognition','data','database.csv')
    data = pd.read_csv(db_path)
    cache_metadata["metadata"] = data


@app.post("/predict")
async def predict_handler(response : Response, inputImage : UploadFile = File(...)):

    '''
    Check extension
    '''
    check = check_extension(inputImage.filename)
    if check == False :
        response_payload = {
                "status" : "error",
                "message" : "Input file format not valid"
                }
        response.status_code=400
        return response

    '''
    Temp image
    '''
    temp_image = str(int(time.time())) + "_" + inputImage.filename
    with open(temp_image, "wb") as buffer:
        shutil.copyfileobj(inputImage.file, buffer)


    '''
    Prediction worker
    '''
    # Extraction image
    img = read_imagefile(temp_image)


    #prediction artiste
    model_artist = cache_models["model_1"]
    print(model_artist)
    pred = model_artist.predict(img)
    artiste_index = np.argmax(pred[0])

    artist_index, artist_name = extract_artist(artiste_index, cache_metadata["metadata"])

    #prediction toile
    model_painting =cache_models["model_3"]
    print(model_painting)
    img_knn = read_imagefile(temp_image)
    layer_outputs = [model_painting.layers[-1].input]
    embedding_model = models.Model(inputs=model_painting.input, outputs=layer_outputs)

    image_embeddings = embedding_model.predict(img_knn)

    model_knn = cache_models["model_2"]
    print(model_knn)

    dist, pred_label = model_knn.kneighbors(image_embeddings.reshape(1,-1),n_neighbors=3,return_distance=True)
    pred_label = list(pred_label[0])

    picture, picture_2, picture_3 = extract_picture(pred_label,cache_metadata["metadata"])

    name, name_2, name_3 = extract_title(pred_label,cache_metadata["metadata"])

    dist, dist_2, dist_3 = extract_distance(dist,cache_metadata["metadata"])

    real_artist, real_artist_2, real_artist_3 = extract_real_artist(pred_label,cache_metadata["metadata"])

    real_artist_name, real_artist_name_2, real_artist_name_3 = extract_real_artist_name(pred_label,cache_metadata["metadata"])



    # response_payload = {"artist_prediction":{"artist_index":artist_index, "artist_name" : artist_name}},\
    # {"url_artists":{"url_artist_index":real_artist,"url_artist_index_2":real_artist_2,"url_artist_index_3":real_artist_3}},\
    # {"url_artist_names":{"url_artist_name":real_artist_name,"url_artist_name_2":real_artist_name_2,"url_artist_name_3":real_artist_name_3}},\
    # {"pictures_predictions":{"picture_number":picture,"picture_number_2":picture_2,"picture_number_3":picture_3}},\
    # {"pictures_names":{'picture_name':name,'picture_name_2':name_2,'picture_name_3':name_3}}
    response_payload = {"artist_index":artist_index, "artist_name" : artist_name,"url_artist_index":real_artist,\
    "url_artist_index_2":real_artist_2,"url_artist_index_3":real_artist_3,"url_artist_name":real_artist_name,\
    "url_artist_name_2":real_artist_name_2,"url_artist_name_3":real_artist_name_3,\
    "picture_number":picture,"picture_number_2":picture_2,"picture_number_3":picture_3,\
    'picture_name':name,'picture_name_2':name_2,'picture_name_3':name_3,\
    'distance':dist,'distance_2':dist_2,'distance_3':dist_3}
    '''
    Delete temp image
    '''
    if os.path.exists(temp_image):
        os.remove(temp_image)


    return response_payload
