from flask import Flask, request, render_template
from elasticsearch import Elasticsearch
import glob
import os
from pathlib import Path
from random import choice,shuffle
from base64 import b64encode

Class_dict = {}
Image_dict = {}
source_folder = Path(os.getenv('QBI_SOURCE',
                               f'{os.path.dirname(os.path.abspath(__file__))}/data/'))

# Prototype function to populate the dictionary based on the files in the
# source folder:
# def load_data():
#     # Create the class doc:
#     for class_avg in glob.glob(os.path.join(source_folder,"*.png")):
#         class_folder = class_avg[:-8]
#         Class_dict[class_folder] = [class_avg,[]]
#         # Create the raw image doc for that class...
#         for image_file in glob.glob(os.path.join(class_folder,"*.png")):
#             image = open(image_file,"rb")
#             contents = image.read()
#             image.close()
#             bin_image = b64encode(contents).decode()
#             Class_dict[class_folder][1].append(image_file)
#             Image_dict[image_file]=[bin_image]
es = Elasticsearch(['http://localhost:9200'])

def load_data():
    # Load raw images in Elasticsearch
    for class_avg in glob.glob(os.path.join(source_folder,"*.png")):
        class_folder = class_avg.replace("_avg.png","")
        Class_dict[class_folder] = [class_avg, []]
        for image_file in glob.glob(os.path.join(class_folder,"*.png")):
            image = open(image_file,"rb")
            contents = image.read()
            image.close()
            bin_image = b64encode(contents).decode()
            Class_dict[class_folder][1].append(image_file.replace('.png',''))
            Image_dict[image_file]=[bin_image,[]]
            es.index('images', 'doc', {"binary": bin_image, "labels":[]}, image_file.replace('.png', ''))

load_data()

app = Flask(__name__)


@app.route('/QBI')
def QBI():
    n=16
    class1 = choice([k for k in Class_dict.keys()] )
    shuffle(Class_dict[class1][1])
    List_names = Class_dict[class1][1][0:int(n)]
    return render_template("QBI.html",
                           Image_list=enumerate(
                               [(id_, es.get('images',
                                             'doc',
                                             id_)['_source']['binary']) for id_ in List_names])
                           )


@app.route('/classify', methods=['POST'])
def classify_image():
    current_dict = request.get_json()
    for image_id, selected in current_dict.items():
        # Retrieve the existing labels for that image using ES:
        image_info = es.get('images', 'doc', image_id)
        # Create
        image_info['_source']['labels'].append(selected)
        es.index('images', 'doc', image_info['_source'], image_id)
    return "success"


# Once the user clicks "Submit", we get these values back,
# and update the dictionary:
def update_entries(image_list,bool_list):
    for image, value in zip(image_list,bool_list):
        print("Incomplete")
        #image_info = es.get('images', 'doc', image_id)
        #image_info['_source']
