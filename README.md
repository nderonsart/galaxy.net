# Space Detector

## Authors

- Nicolas Deronsart
- Clément Bonduelle

![Logo](logo.jpg "Logo of Space-Detector")

## Description

This is the final project in the Big Data module, the objective is to, to a given picture, tell if there is a galaxy.
The pictures for the model training will be retrieved from [Hubble Website](https://esahubble.org/images/), we will only use Hubble pictures to ensure picture quality.

In this website, pictures are labelled in the `Object description` field. The training dataset will be constructed by labelling whether if the `Object description` contains the word `galaxy` or not.

The model will be served on an online website that will allow to upload a picture and get a response telling if a galaxy is in it or not.

## Objective

The objective of this project is to create a collaborative website `Galaxy.net` where people can add pictures of galaxies. To ensure that the pictures are relevant, we use a model to predict if they contain a galaxy or not.  

## Project structure

The project is structured as follows:

```
.
├── data
│   ├── labelled_pictures.csv
│   └── ...
├── models
│   └── galaxy_classifier-v{...}.pt
├── notebooks
│   ├── model_design.ipynb
│   └── web_scraping.ipynb
├── src
│   ├── model_training
│   │   ├── CNN.py
│   │   └── train_model.py
│   └── web_scraping.py
├── tests
│   ├── test.jpg
│   ├── test2.jpg
│   └── test_model.py
├── .gitignore
├── logo.jpg
├── README.md
├── requirements.txt
└── scrap_and_train.sh
```

- `data`: contains the data used for training.
    - `labelled_pictures.csv`: contains the labelled pictures.
    - `jpg files`: the pictures in jpg format.
- `models`: contains the trained models, using the format `galaxy_classifier-v{...}.pt` where `{...}` is the version of the model.
- `notebooks`: contains the notebooks used for the project.
    - `model_design.ipynb`: the notebook used to design the model.
    - `web_scraping.ipynb`: the notebook used to design the data scraping.
- `src`: contains the code source of the project.
    - `model_training`: contains the code used to train the model.
        - `CNN.py`: the model architecture.
        - `train_model.py`: the script used to train the model on the labelled data.
    - `web_scraping.py`: the module used to scrap the data.
- `tests`: contains the tests for the project.
    - `test_model.py`: the tests for the model.
    - `test.jpg` : picture of the Moon used for testing. (Picture not used for training, retrieved from [Envydya's Instagram](https://www.instagram.com/envydya/))
    - `test2.jpg` : picture of a Galaxy used for testing. (Picture not used for training, retrieved from [Envydya's Instagram](https://www.instagram.com/envydya/))
- `.gitignore`: the gitignore file.
- `logo.jpg` : logo of the project.
- `README.md`: the readme file.
- `requirements.txt`: the requirements file.

## Installation

To use this project, you can create a venv using : 
```
python -m venv env
```

Then you can activate it using the following command on Unix : 
```
env/bin/activate
```

Or if you are on Windows :
```
env\Scripts\activate.bat
``` 

And finally install the dependencies using 
```
pip install -r requirements.txt
```

## Get the data and train the model

### Unix
To get the data and train the model, you can run the `scrap_and_train.sh` script using :
```
./scrap_and_train.sh
```
It will run the `web_scraping.py` script and then the `train_model.py` script.
If it doesn't work, you should run : 
```
chmod +x scrap_and_train.sh
```

### Windows

You can simply run the following two commands :

```
python src/web_scraping.py
python src/model_training/train_model.py
```

It will train the model on the data in the `data` folder (49 images from the Hubble website retrieved by `web_scraping.py` with a csv file containing a labelling) and save the model in the `models` folder. The model will be saved using the format `galaxy_classifier-v{...}.pt` where `{...}` is the version of the model.

## The application

The application consists of a website where you can upload a picture and get a response telling if a galaxy is in it or not. If the picture contains a galaxy, the website will automatically add it to the main page of the website. It is composed of three parts : the database, the api and the webapp.

### The database

The database is a MongoDB database that contains the pictures of the galaxies.
You have to create a database named `galaxy` and a collection named `images` to use the application.
It will automatically add the pictures uploaded by the users to the database.

### The API

The API is a FastAPI application that uses the model to predict if a picture contains a galaxy or not. It has one endpoint :
- `/predict`: it takes pictures in the list format as inputs and returns a json response with the prediction.
- `/images`: it returns the list of the galaxy images from the database.

To run the API, you can run the following commands from the root folder : 
```
python src/api/app.py
command uvicorn src.api.app:app --host 0.0.0.0 --reload
```
Don't forget to setup Mongo environment variable with for example : 
```
URL_MONGO='mongodb://admin:admin@localhost:27017'
```

While running the API is available at the URL : 
```
http://127.0.0.1:8000
```

### The webapp

The webapp is a Streamlit application with two pages :
- `Images`: the main page of the website, it contains the pictures of the galaxies given by the users.
- `Upload new image`: the page where you can upload a picture and get a response telling if a galaxy is in it or not.

To run the webapp, you can run the following command from the root folder : 
```
API_URL='http://127.0.0.1:8000' streamlit run src/webapp/app.py
```

The website will be available in local at the URL : 
```
http://localhost:8501
```

## Testing

To run tests, you have to run the following command (make sure that the API is up before you run it):
```
API_URL='http://127.0.0.1:8000' python tests/test_model.py
```
With API_URL the URL of the API.