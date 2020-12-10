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

def check_extension(filename):
    ALLOWED_EXTENSION = ["jpg", "jpeg", "png"]
    # Extract extension
    extension = filename.split(".")[-1:][0].lower()
    if extension not in ALLOWED_EXTENSION :
        return False
    else :
        return True

def read_imagefile(file) -> Image.Image:
    #img = Image.open(BytesIO(file))
    img = Image.open(file)
    img = img.resize((224,224),resample=Image.BILINEAR)
    img = np.array(img)
    img = np.expand_dims(img, axis = 0)
    return img

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
    Prediction worker
    '''
    # Extraction image
    #img = read_imagefile( await inputImage.read())
    img = read_imagefile(inputImage.file)

    #prediction
    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname,'models','Model_v0_art12_emb100_withDA')
    model = load_model(model_path)

    pred = model.predict(img)
    artiste_index = np.argmax(pred[0])
    print(artiste_index)

    response_payload = {"prediction" : str(artiste_index)}

    return response_payload
