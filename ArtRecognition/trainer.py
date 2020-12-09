import os
import time
import numpy as np
import pandas as pd
#from memoized_property import memoized_property
#import mlflow
#from mlflow.tracking import MlflowClient
from google.cloud import storage
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, Sequential, Input, Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.data.experimental import cardinality
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

BUCKET_NAME = 'art-recognition-app'
BUCKET_TRAIN_DATA_PATH = 'Top_12_artists'
MODEL_NAME = 'VGG16'
MODEL_VERSION = 'v1'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
path0 = r'C:\Users\pitip\code\ClaireLeroux44\ArtRecognition'

class Trainer(object):
    def __init__(self, n_embedding=100, data_aug=False, run_local=False, **kwargs):
        self.n_embedding = n_embedding
        self.data_aug = data_aug
        self.run_local = run_local

    def get_datasets(self):
        print("Prepare datasets loading:")
        if run_local:
            self.train_dir = os.path.join(path0, 'raw_data', 'test_VGG16', 'Train')
            self.test_dir = os.path.join(path0, 'raw_data', 'test_VGG16', 'Test')
        else:
            client = storage.Client()

            self.train_dir = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}/Train"
            self.test_dir = f"gs://{BUCKET_NAME}/{BUCKET_DATA_PATH}/Test"


        train_dataset = image_dataset_from_directory(self.train_dir, shuffle=True, batch_size=BATCH_SIZE,\
            image_size=IMG_SIZE, label_mode='categorical')
        test_dataset = image_dataset_from_directory(self.test_dir, shuffle=True, batch_size=BATCH_SIZE, \
            image_size=IMG_SIZE, label_mode='categorical')

        train_batches = cardinality(train_dataset)
        validation_dataset = train_dataset.take(train_batches // 5)
        train_train_dataset = train_dataset.skip(train_batches // 5)

        self.train_train_dataset = train_train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        self.n_artist = len(train_dataset.class_names)

        print(f'Number of detected classes: {self.n_artist}')
        print(f'Number of train/val/test batches: {cardinality(train_train_dataset)}/{cardinality(validation_dataset)}/{cardinality(test_dataset)}')

    def get_model(self):
        print("Define model")
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        base_model.trainable = False

        model_layers = [base_model, layers.Flatten(),
                      layers.Dense(self.n_embedding, activation='relu')]
        model1 = models.Sequential(layers = model_layers)

        data_augmentation = Sequential([layers.experimental.preprocessing.RandomFlip('horizontal'), \
                                layers.experimental.preprocessing.RandomRotation(0.1),\
                                layers.experimental.preprocessing.RandomContrast(0.1),\
                                layers.experimental.preprocessing.RandomZoom(height_factor=0.2), \
                                layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2)\
])

        inputs = Input(shape=(224, 224, 3))
        if self.data_aug == True:
            print("Add data augmentation layer")
            x = data_augmentation(inputs)
            x = preprocess_input(x)
        else:
            x = preprocess_input(inputs)
        x = model1(x)
        prediction_layer = layers.Dense(self.n_artist, activation='softmax')
        outputs = prediction_layer(x)
        self.model = Model(inputs, outputs)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    def model_train(self):
        tic = time.time()
        self.get_datasets()

        self.get_model()

        print("Fit model")
        self.model.fit(self.train_train_dataset, epochs=2, validation_data=self.test_dataset, batch_size=32,
                    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)])
        elapsed = time.time() - tic
        print(f"Elapsed time: {elapsed:.2f} s")

    def save_model(self, model_iter=0):
        """Save the model"""

        storage_name= f"models/{MODEL_NAME}_{MODEL_VERSION}_{model_iter}"

        save_model(self.model, storage_name, overwrite=True, include_optimizer=True)
        print(f"saved {storage_name} locally")

        if run_local == False:
            storage_location = BUCKET_NAME
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(storage_location)
            blob = bucket.blob(storage_name)
            blob.upload_from_filename(storage_name)
            print(f"saved {storage_name} on GS")

    def evaluate_model(self):
        self.evaluation_loss, self.evaluation_accuracy = self.model.evaluate(self.test_dataset)
        print('Test accuracy :', self.evaluation_accuracy)

    # @memoized_property
    # def mlflow_client(self):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     return MlflowClient()

    # @memoized_property
    # def mlflow_experiment_id(self):
    #     try:
    #         return self.mlflow_client.create_experiment(self.experiment_name)
    #     except BaseException:
    #         return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # @memoized_property
    # def mlflow_run(self):
    #     return self.mlflow_client.create_run(self.mlflow_experiment_id)

    # def mlflow_log_param(self, key, value):
    #     self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    # def mlflow_log_metric(self, key, value):
    #     self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # def log_estimator_params(self):
    #     reg = self.get_estimator()
    #     self.mlflow_log_param('estimator_name', reg.__class__.__name__)
    #     params = reg.get_params()
    #     for k, v in params.items():
    #         self.mlflow_log_param(k, v)

    # def log_kwargs_params(self):
    #     if self.mlflow:
    #         for k, v in self.kwargs.items():
    #             self.mlflow_log_param(k, v)

    # def log_machine_specs(self):
    #     cpus = multiprocessing.cpu_count()
    #     mem = virtual_memory()
    #     ram = int(mem.total / 1000000000)
    #     self.mlflow_log_param("ram", ram)


if __name__=="__main__":
    run_local = False

    n_embedding=100
    print("Instanciate trainer")
    trainer = Trainer(n_embedding=n_embedding, data_aug=False, run_local=run_local)
    print("Training")
    trainer.model_train()

    print("Save model")
    trainer.save_model(model_iter=1)

    print("Evaluate model")
    trainer.evaluate_model()
