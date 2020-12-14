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
    return artist_name

def extract_picture(pred_label,metadb):
    pics_list = []
    for i in pred_label[0]:
        df = metadb[metadb['labels'] == i]
        pics_list.append(list(df['pics'])[0])
    return pics_list[0]


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
    return artist_name

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
    model_path = os.path.join(dirname,'models','model_v5')
    model_artist = load_model(model_path)
    cache_models["model_1"] = model_artist

    print("loading model knn... ")
    dirname = os.path.dirname(os.path.dirname(__file__))
    knn_path = os.path.join(dirname,'models','KNN_models','model2.joblib')
    model_knn = joblib.load(knn_path)
    cache_models["model_2"] = model_knn

    print("loading metadata database ... ")
    dirname = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(dirname,'ArtRecognition','data','database.csv')
    data = pd.read_csv(db_path)
    cache_metadata["metadata"] = data


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
    pred = model_artist.predict(img)
    artiste_index = np.argmax(pred[0])

    artist_name = extract_artist(artiste_index, cache_metadata["metadata"])

    #prediction toile
    img_knn = read_image_knn(temp_image)
    layer_outputs = [model_artist.layers[-1].input]
    embedding_model = models.Model(inputs=model_artist.input, outputs=layer_outputs)

    image_embeddings = embedding_model.predict(img_knn)

    model_knn = cache_models["model_2"]
    print(model_knn)

    dist, pred_label = model_knn.kneighbors(image_embeddings.reshape(1,-1),n_neighbors=3,return_distance=True)

    picture = extract_picture(pred_label,cache_metadata["metadata"])



    response_payload = {"artist_prediction" : artist_name,"picture_name":picture}
    #response_payload = {"prediction" : picture}
    '''
    Delete temp image
    '''
    if os.path.exists(temp_image):
        os.remove(temp_image)

    '''
    Delete temp image
    '''
    if os.path.exists(temp_image):
        os.remove(temp_image)
    return response_payload
