#marie: code pour création API
#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, Response
import json
import numpy as np
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from PIL import Image
from io import BytesIO
import os
import pandas as pd 
import time
import shutil




app = FastAPI()

models = {}
cache_metadata = {}

def check_extension(filename):
    ALLOWED_EXTENSION = ["jpg", "jpeg", "png"]
    # Extract extension
    extension = filename.split(".")[-1:][0].lower()
    if extension not in ALLOWED_EXTENSION :
        return False
    else :
        return True

def path_to_image(input_img, image_size, num_channels, interpolation):
# img = io_ops.read_file(path)
    img = image_ops.decode_image(input_img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

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

# def read_imagefile(file) -> Image.Image:
# img = Image.open(BytesIO(file))
# img = Image.open(file)
# print(type(img))
# img_test = path_to_image(img, (224, 224), 3, 'bilinear')
# img_test = np.array(img_test)
# img_test = np.expand_dims(img_test, axis = 0)
# return img_test

# rgbimg.save('foo.jpg')

# def read_imagefile(file) -> Image.Image:
# img = Image.open(file)
# img = img.resize((224,224),resample=Image.BILINEAR)

# rgbimg = Image.new("RGBA", img.size)
# rgbimg.paste(img)

# rgbimg = np.array(rgbimg)
# rgbimg = np.expand_dims(rgbimg, axis = 0)
# return rgbimg

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

@app.on_event("startup")
async def startup_event():
    print("loading model")
    model = "my_model"
    models["model_1"] = model

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
        #img = read_imagefile( await inputImage.read())
    img = read_imagefile(temp_image)

    '''
    Delete temp image
    '''
    if os.path.exists(temp_image): 
        os.remove(temp_image)


    #prediction
# dirname = os.path.dirname(os.path.dirname(__file__))
# print(models["model_1"])

    artiste_index = 12

    response_payload = {"prediction" : str(artiste_index)}

    return response_payload["prediction"]
