import numpy as np
import pandas as pd

import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch

import matplotlib.pyplot as plt

import os

from CNN import CNN, train


SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_img(img_name):
    """
    Load an image from the data folder
    parameters:
        - img_name: name of the image file
    returns:
        - the image as a numpy array
    """
    return plt.imread('data/' + img_name)


def compress_img(img, width=300, height=300):
    """
    Compress an image to a given size
    parameters:
        - img: image to compress
        - width: width of the compressed image
        - height: height of the compressed image
    returns:
        - the compressed image
    """
    img = cv2.resize(img, (width, height))
    return img


def data_augmentation(img):
    """
    Create new images from an existing one by applying transformations
    parameters:
        - img: image to transform
    returns:
        - a tuple of transformed images
    """
    return (cv2.flip(img, 1), cv2.flip(img, 0), cv2.flip(img, -1),
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(img, cv2.ROTATE_180))


def create_Xs_ys(df):
    """
    Create the X and y arrays from the dataframe
    parameters:
        - df: dataframe containing the data
    returns:
        - X_train: training images
        - X_test: testing images
        - y_train: training labels
        - y_test: testing labels
    """
    X, y = [], []
    for i, row in df.iterrows():
        img = load_img(row['file'])
        img = compress_img(img)
        X.append(img)
        y.append(row['galaxies'])
    X, y = np.array(X), np.array(y)

    X_aug, y_aug = [], []
    for i, img in enumerate(X):
        X_aug.append(img)
        y_aug.append(y[i])
        for img_aug in data_augmentation(img):
            X_aug.append(img_aug)
            y_aug.append(y[i])
    X_aug, y_aug = np.array(X_aug), np.array(y_aug)

    return train_test_split(
        X_aug, y_aug, random_state=SEED)


if __name__ == '__main__':
    df = pd.read_csv('data/labelled_pictures.csv', sep=';')
    df['galaxies'] = df['target'].apply(lambda x: 1 if x == 'Galaxies' else 0)

    X_train, X_test, y_train, y_test = create_Xs_ys(df)

    model = CNN()

    train(model, X_train, y_train, X_test, y_test,
          epochs=25, batch_size=32, lr=0.001)

    acc = accuracy_score(y_test, np.round(
        torch.sigmoid(
            model(torch.from_numpy(
                X_test).float().permute(0, 3, 1, 2))).detach()))

    print(f'Accuracy: {acc}')

    version = len(os.listdir('models')) + 1
    torch.save(model.state_dict(),
               f'models/galaxy_classifier-v{version}.pt')
