from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import models
from keras import layers
from keras import optimizers
import argparse
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os
import shutil
train_data="./data/train"
val_data="./data/validation"
sim_data="./sim_data/"


class model_set:
    def __init__(self,nTrain,nVal,batch_size=20,sim_multiplier=15):
        self.vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        self.nTrain=nTrain
        self.nVal=nVal
        self.batch_size=batch_size
        self.sim_multiplier=sim_multiplier

        self.train_features = np.zeros(shape=(self.nTrain*self.sim_multiplier, 7, 7, 512))
        self.train_labels = np.zeros(shape=(self.nTrain*self.sim_multiplier,2))

        self.validation_features = np.zeros(shape=(self.nVal, 7, 7, 512))
        self.validation_labels = np.zeros(shape=(self.nVal,2))


    def freeze(self):
        for layer in self.vgg_conv.layers[:-4]:
            layer.trainable = False

    def freeze_status(self):
        for layer in self.vgg_conv.layers:
            print(layer, layer.trainable)

    def add_layers(self):
        # Create the model
        #self.model = models.Sequential()
        # Add the vgg convolutional base model
        #self.model.add(self.vgg_conv)
        # Add new layers
        #self.model.add(layers.Flatten())
        #self.model.add(layers.Dense(1024, activation='relu'))
        #self.model.add(layers.Dropout(0.5))
        #self.model.add(layers.Dense(3, activation='softmax'))
        self.model = models.Sequential()
        self.model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(2, activation='softmax'))
        self.model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

    def extract_train_features(self):
        print("\nTRAINING:")
        datagen = ImageDataGenerator(rotation_range=180)
        self.train_generator = datagen.flow_from_directory(
            sim_data,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle='shuffle')      
        
        

        nImages=self.nTrain*self.sim_multiplier
        i = 0
        for inputs_batch, labels_batch in self.train_generator:
            print("Processing batch {}/{}".format(i * self.batch_size,self.sim_multiplier*self.nTrain),end="\r",flush=True)
            features_batch = self.vgg_conv.predict(inputs_batch)
            self.train_features[i * self.batch_size : (i + 1) * self.batch_size] = features_batch
            self.train_labels[i * self.batch_size : (i + 1) * self.batch_size] = labels_batch
            i += 1
            if i * self.batch_size >= nImages:
                break
        print("Processing batch {}/{}".format((i) * self.batch_size,self.sim_multiplier*self.nTrain),flush=True)
                    
        self.train_features = np.reshape(self.train_features, (self.nTrain*self.sim_multiplier, 7 * 7 * 512))      

    def extract_validation_features(self):
        print('\nVALIDATION')
        datagen = ImageDataGenerator()
        self.validation_generator = datagen.flow_from_directory(
            val_data,
            target_size=(224, 224),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle='shuffle')
        nImages=self.nVal
        i = 0
        for inputs_batch, labels_batch in self.validation_generator:
            print("Processing batch {}/{}".format(i * self.batch_size,self.nVal),end="\r",flush=True)
            features_batch = self.vgg_conv.predict(inputs_batch)
            self.validation_features[i * self.batch_size : (i + 1) * self.batch_size] = features_batch
            self.validation_labels[i * self.batch_size : (i + 1) * self.batch_size] = labels_batch
            i += 1
            if i * self.batch_size >= nImages:
                break
        print("Processing batch {}/{}".format((i) * self.batch_size,self.nVal),flush=True)
        self.validation_features = np.reshape(self.validation_features, (self.nVal, 7 * 7 * 512))
    
    def get_history(self,epochs=40):
        self.history = self.model.fit(self.train_features,
                    self.train_labels,
                    epochs=40,
                    batch_size=self.batch_size,
                    validation_data=(self.validation_features,self.validation_labels))

    def summary(self):
        fnames = self.validation_generator.filenames
 
        ground_truth = self.validation_generator.classes
         
        label2index = self.validation_generator.class_indices
         
        # Getting the mapping from class index to class label
        idx2label = dict((v,k) for k,v in label2index.items())
         
        predictions = self.model.predict_classes(self.validation_features)
        prob = self.model.predict(self.validation_features)
        #print(predictions)
        #print(ground_truth)
        #errors = np.where(predictions != ground_truth)[0]
        #print("No of errors = {}/{}".format(len(errors),nVal))
        
        
        #print(errors)
        wrong=0
        for i in range(len(predictions)):
            if predictions[i]!=ground_truth[i]:
                pred_class = np.argmax(prob[i])
                pred_label = idx2label[pred_class]
                 
                print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
                    fnames[i].split('/')[0],
                    pred_label,
                    prob[i][pred_class]))
                print(prob[i])
                original = mpimg.imread('{}/{}'.format(val_data,fnames[i]))
                wrong+=1
                #plt.imshow(original)
                #plt.show()
        print("Incorrect: {}/{}".format(wrong,len(predictions)))

def build_simmed_images(mod,verbose=False):
    if verbose: print("Making Simulated Images")
    #if directory exists, remove it and make again
    if not os.path.isdir(sim_data):os.mkdir(sim_data)
    if os.path.isdir(os.path.join(sim_data,"matched")): shutil.rmtree(os.path.join(sim_data,"matched"))
    os.mkdir(os.path.join(sim_data,"matched"))

    if os.path.isdir(os.path.join(sim_data,"mismatched")): shutil.rmtree(os.path.join(sim_data,"mismatched"))
    os.mkdir(os.path.join(sim_data,"mismatched"))
    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=False,
        fill_mode='nearest')
    #find all images
    image_set=os.listdir(os.path.join(train_data,"matched"))
    for f in image_set[:mod.nTrain]:
        img = load_img(os.path.join(train_data,"matched",f))  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        shutil.copy(os.path.join(train_data,"matched",f), os.path.join(sim_data,"matched"))
    
        for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=os.path.join(sim_data,"matched"), save_prefix="simm", save_format='png'):
            i += 1
            if i > mod.sim_multiplier:
                break  # otherwise the generator would loop indefinitely
            pass
    image_set=os.listdir(os.path.join(train_data,"mismatched"))
    for f in image_set[:mod.nTrain]:
        img = load_img(os.path.join(train_data,"mismatched",f))  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0

        shutil.copy(os.path.join(train_data,"mismatched",f), os.path.join(sim_data,"mismatched"))
        for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=os.path.join(sim_data,"mismatched"), save_prefix="simm", save_format='png'):
            i += 1
            if i > mod.sim_multiplier:
                break  # otherwise the generator would loop indefinitely
            pass
    

def extract_features():
    pass

def fit_data():
    pass

def main(verbose=False):
    #Set up the network
    mod=model_set(100,400)
    #mod.freeze()
    #if verbose: mod.freeze_status()
    mod.add_layers()
    #if verbose: mod.model.summary()
    
    #Create the simmed data
    build_simmed_images(mod,verbose=verbose)

    mod.extract_train_features()
    mod.extract_validation_features()
    mod.get_history()
    mod.summary()


if __name__=="__main__":
    main()
