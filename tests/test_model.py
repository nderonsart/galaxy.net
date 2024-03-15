import os
import sys
import requests

API_URL = os.environ.get('API_URL')
MODEL_PATH = "models/galaxy_classifier-v"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training.CNN import CNN
import torch
import unittest
import matplotlib.pyplot as plt
import cv2
import numpy as np


class TestMLModel(unittest.TestCase):
    """
    Class to test our model using unittest
    """
    def setUp(self) -> None:
        """
        Setup our tests
        """
        self.model = CNN()
        self.model.load_state_dict(torch.load(MODEL_PATH +
                                 f"{len(os.listdir('models'))}.pt"))
        print("\n______________________\nStart of test\n")

    def tearDown(self) -> None:
        """
        Print after test in terminal
        """
        print("\nEnd of test\n______________________")

    def test_model(self) -> None:
        """
        Checks if the model is a CNN model
        """
        print("[Test] CHECK IF MODEL IS CNN")
        self.assertEqual(type(self.model), CNN)

    def test_upload_connection(self) -> None:
        """
        Check if we have a response from the server
        """
        print("[Test] CONNECTION TO API")
        img = plt.imread("tests/test.jpg")
        img_resized = cv2.resize(img, (300, 300))
        response = requests.post(f'{API_URL}/predict',
                                     json={"images": [img_resized.tolist()]})
        self.assertEqual(response.status_code, 200)

    def test_upload_coherence(self) -> None:
        """
        Checks if the response from the server is correct
        """
        print("[Test] API RESPONSE IS CORRECT")
        img = plt.imread("tests/test.jpg")
        img_resized = cv2.resize(img, (300, 300))
        response = requests.post(f'{API_URL}/predict',
                                     json={"images": [img_resized.tolist()]})
        self.assertEqual(response.json()["galaxies"][0], "No")

    def test_predict_response_type(self) -> None:
        """
        Checks if the prediction of an image of the Moon is "No" or "Yes"
        """
        print("[Test] PREDICT LOCALLY")
        img = plt.imread("tests/test.jpg")
        img_resized = cv2.resize(img, (300, 300)).tolist()
        input = np.array([img_resized])
        y_pred = self.model(torch.from_numpy(np.array(
            input, dtype=np.int32)).float().permute(0, 3, 1, 2))
        y_pred = np.round(torch.sigmoid(y_pred.detach()))
        y_pred = ['No' if y == 0 else 'Yes' for y in y_pred]
        self.assertIn(y_pred[0], ["No", "Yes"])

    def test_predict_response_coherence_Moon(self) -> None:
        """
        Checks if the prediction of an image of the Moon is "No"
        """
        print("[Test] PREDICT COHERENCE LOCALLY ON MOON")
        img = plt.imread("tests/test.jpg")
        img_resized = cv2.resize(img, (300, 300)).tolist()
        input = np.array([img_resized])
        y_pred = self.model(torch.from_numpy(np.array(
            input, dtype=np.int32)).float().permute(0, 3, 1, 2))
        y_pred = np.round(torch.sigmoid(y_pred.detach()))
        y_pred = ['No' if y == 0 else 'Yes' for y in y_pred]
        self.assertEqual(y_pred[0], "No")

    def test_predict_response_coherence_Galaxy(self) -> None:
        """
        Checks if the prediction of an image of a Galaxy is "Yes"
        """
        print("[Test] PREDICT COHERENCE LOCALLY ON GALAXY")
        img = plt.imread("tests/test2.jpg")
        img_resized = cv2.resize(img, (300, 300)).tolist()
        input = np.array([img_resized])
        y_pred = self.model(torch.from_numpy(np.array(
            input, dtype=np.int32)).float().permute(0, 3, 1, 2))
        y_pred = np.round(torch.sigmoid(y_pred.detach()))
        y_pred = ['No' if y == 0 else 'Yes' for y in y_pred]
        self.assertEqual(y_pred[0], "Yes")
    

if __name__ == '__main__':
    unittest.main()
