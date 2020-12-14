#marie: code pour création API
#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, Response
import json
import numpy as np
from PIL import Image
from io import BytesIO
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import pandas as pd


app = FastAPI()


cache_models = {}

def check_extension(filename):
    ALLOWED_EXTENSION = ["jpg", "jpeg", "png"]
    # Extract extension
    extension = filename.split(".")[-1:][0].lower()
    if extension not in ALLOWED_EXTENSION :
        return False
    else :
        return True

# def path_to_image(path, image_size, num_channels, interpolation):
#     img = io_ops.read_file(path)
#     img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)
#     img = image_ops.resize_images_v2(img, image_size, method=interpolation)
#     img.set_shape((image_size[0], image_size[1], num_channels))
#     return img

def read_imagefile(file) -> Image.Image:
    img = Image.open(file)
    img = img.resize((224,224),resample=Image.BILINEAR)
    rgbimg = Image.new("RGB", img.size)
    rgbimg.paste(img)
    rgbimg = np.array(rgbimg)
    rgbimg = np.expand_dims(rgbimg, axis = 0)
    print(rgbimg.shape)
    return rgbimg


@app.on_event("startup")
async def startup_event():
    print("loading model ...")
    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname,'models','model_v5')
    model = load_model(model_path)
    cache_models["model_1"] = model

@app.post("/predict")
async def predict_handler(response : Response, inputImage : UploadFile = File(...)):

    '''
    Check extension
    '''
    check = check_extension(inputImage.filename)
    print(check)
    if check == False :
        response_payload = {
                "status" : "error",
                "message" : "Input file format not valid"
                }
        response.status_code=400
        return response

    '''
    Prediction worker
    '''
    # Extraction image
    #img = read_imagefile( await inputImage.read())
    img = read_imagefile(inputImage.file)


    #prediction
    model = cache_models["model_1"]
    pred = model.predict(img)
    artiste_index = np.argmax(pred[0])
    print(artiste_index)
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
    dirname = os.path.dirname(os.path.dirname(__file__))
    db_path = os.path.join(dirname,'ArtRecognition','data','database.csv')
    data = pd.read_csv(db_path)
    data_artist =data[data.loc[:,'artist_number'] == dico_artistes[artiste_index]]
    artist_name = list(data_artist['artist'])[0]

    response_payload = {"prediction" : artist_name}

    return response_payload["prediction"]
