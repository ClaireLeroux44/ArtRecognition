import os
import numpy as np
import matplotlib.pyplot as plt
from google.cloud import storage
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
from trainer import BUCKET_NAME, BUCKET_DATA_PATH


def download_model(model_name, bucket=BUCKET_NAME, run_local=True):
    if run_local==False:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_name}")

        print(storage_location)
        blob = client.blob(storage_location)
        blob.download_to_filename(model_name)
        print("=> model downloaded from storage")
    model = load_model(f"models/{model_name}", custom_objects=None, compile=False)

    print("=> model loaded")
    return model

def artist_prediction(model, image):
    pred = model.predict(image)
    proba = pred[0]
    artiste_index = np.argmax(pred[0])
    return artiste_index, proba


if __name__ == '__main__':
    model_name = r'VGG16_v1_1'
    model = download_model(model_name, run_local=True)

    image_to_predict =r'C:\Users\pitip\code\ClaireLeroux44\ArtRecognition\raw_data\test_VGG16\Test\class_3\118.jpg'

    im_224 = load_img(image_to_predict, grayscale=False, color_mode='rgb', target_size=(224, 224), interpolation='bilinear')
    im_224 = np.array(im_224.getdata()).reshape(im_224.size[0], im_224.size[1], 3)
    im_224 = np.expand_dims(im_224, axis = 0)

    artiste_index_pred, proba_pred = artist_prediction(model, im_224)
    print(f"Predicted artist index: {artiste_index_pred}")
    print(f"Predicted probabilities: {proba_pred}")
