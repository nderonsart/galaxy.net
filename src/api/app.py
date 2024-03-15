import numpy as np
import torch

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.fields import Field

from loguru import logger

import pymongo

import os

import sys
sys.path.append('src/model_training')
from CNN import CNN

MODEL_PATH = "models/galaxy_classifier-v"

URL_MONGO = os.getenv('URL_MONGO')


client = pymongo.MongoClient(URL_MONGO)
db = client["galaxies"]
collection = db["images"]


app = FastAPI(title='Galaxy Classifier API', api='v1', version='1.0.0')


class PredictInput(BaseModel):
    images: list[list[list[list[int]]]] = Field(description='List of images')


model = CNN()
model.load_state_dict(torch.load(MODEL_PATH +
                                 f"{len(os.listdir('models'))}.pt"))


@app.post('/predict',
          summary='Prediction of the presence of a galaxy in the images')
def predict(input: PredictInput):
    '''
    Prediction of the presence of a galaxy in the images
    parameters:
        - input: input data (list of images under the list format)
    returns:
        - list of predictions
    '''
    logger.info('Predicting galaxy presence')
    try:
        input = input.dict()['images']
        input = np.array(input)
        y_pred = model(torch.from_numpy(np.array(
            input, dtype=np.int32)).float().permute(0, 3, 1, 2))
        y_pred = np.round(torch.sigmoid(y_pred.detach()))
        logger.debug(f'Prediction of galaxy presence: {y_pred}')

        y_pred = ['No' if y == 0 else 'Yes' for y in y_pred]

        for y in y_pred:
            if y == 'Yes':
                collection.insert_one({'image': input[0].tolist()})

        return {
            'galaxies': y_pred
        }
    except Exception as e:
        logger.error(e)
        raise e


@app.get('/images',
         summary='Get all images')
def get_images():
    '''
    Get all images
    returns:
        - list of images
    '''
    logger.info('Getting images')
    try:
        images = collection.find()
        images = [image['image'] for image in images]
        return {
            'images': images
        }
    except Exception as e:
        logger.error(e)
        raise e
