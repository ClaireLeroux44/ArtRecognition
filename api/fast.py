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

def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def read_imagefile(file) -> Image.Image:
    #img = Image.open(BytesIO(file))
    img_test = path_to_image(file, (224, 224), 3, 'bilinear')
    img_test = np.array(img_test)
    img_test = np.expand_dims(img_test, axis = 0)
    return img_test

@app.on_event("startup")
async def startup_event():
    print("loading model ...")
    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname,'models','20201210_170338_VGG16_v2_0')
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

    response_payload = {"prediction" : str(artiste_index)}

    return response_payload
