import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


SITE = 'https://esahubble.org/images/'
NB_IMGS = 49


def find_pictures():
    """
    Find the 50 links to the picture on the first page from Hubble's website
    and find the sky object category
    returns:
        - img_urls: Direct download links to the picture
        - img_categ: Category of the picture (Galaxy, Nebulae, Solar...)
    """
    response = requests.get(SITE)

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    urls = [img["src"] for img in img_tags]

    images = soup.find_all("script")[0].text.split("url")[1:]
    urls = [SITE+url.split("'")[1].split("/images/")[1]
            for url in images if url.__contains__("jpg")]

    img_urls = []
    img_categ = []
    for url in urls[:NB_IMGS]:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tags = soup.find_all('img')[2]["src"]
        categorie = soup.find("div",
                              {"class": "object-info"}).find_all("a")[-1].text
        img_urls.append(img_tags)
        img_categ.append(categorie)
    return img_urls, img_categ


def create_data(img_urls, img_categ):
    """
    Retrieve the pictures from the website and create the data filed that will
    be used for training
    parameters:
        - img_urls: Direct download links to the picture
        - img_categ: Category of the picture (Galaxy, Nebulae, Solar...)
    """
    data = pd.DataFrame(columns=["file", "target"])
    for idx in range(len(img_urls)):
        filename = re.search(r'/([\w_-]+[.](jpg|gif|png))$', img_urls[idx])
        if not filename:
            print("Regex didn't match with the url: {}".format(img_urls[idx]))
            continue
        with open("data/"+filename.group(1), 'wb') as f:
            response = requests.get(img_urls[idx])
            f.write(response.content)
            curr = pd.DataFrame([[filename.group(1), img_categ[idx]]],
                                columns=["file", "target"])
            data = pd.concat([curr, data])

    data.to_csv("data/labelled_pictures.csv", sep=";", index=False)


if __name__ == '__main__':
    img_urls, img_categ = find_pictures()
    create_data(img_urls, img_categ)
