from flask import Flask
from elasticsearch import Elasticsearch
import glob
import os
from pathlib import Path

app = Flask(__name__)
es = Elasticsearch()
source_folder = Path(os.getenv('QBI_SOURCE',
                               f'{os.path.dirname(os.path.abspath(__file__))}/data/'))


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/image/{image_id}', methods=['GET'])
def get_image(image_id):
    image_info = es.get('images', 'doc', image_id)['_source']
    return image_info


@app.route('/classify/{image_id}')
def classify_image(image_id):
    return "Image Classify"


# Prototype function to populate the dictionary based on the files in the
# source folder:
def load_dict():
    Image_dict = {}
    for image_file in glob.glob(os.join(source_folder),"*.png"):
        Image_dict[image_file]=[]
        print(image_file)
    return Image_dict

# Once the user clicks "Submit", we get these values back,
# and update the dictionary:
def update_entries(image_list,bool_list):
    for image, value in zip(image_list,bool_list):
        image_info = es.get('images', 'doc', image_id)
        image_info['_source']

Image_Dict = load_dict()
