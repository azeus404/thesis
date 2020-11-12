import os
import numpy as np
import argparse
from keras.optimizers import Adam
from keras.models import load_model
from keras import Model, Input, Sequential
from keras.layers import Conv2D,GlobalAveragePooling2D,Convolution2D,MaxPool2D
from keras.layers import Dropout, Activation, Dense, Flatten,MaxPooling2D,BatchNormalization
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

"""
A simple implementation of the trained MobileNet models.
This detector can detect and identify a ransomware family from a image.

"""
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, type=str, default="MobileNet",
	help="name of pre-trained network to use")
ap.add_argument("-i", "--image", required=True, type=str, default="f266ee43527fc6258dfec7dc71eadc93711f70a53881270e4d7cc379e4416f77.png",
	help="Image to test on")
args = vars(ap.parse_args())

path = os.path.abspath(os.getcwd())

def MobileNetmodel(no_classes,shape):
    base_model = MobileNet(include_top=False, weights='imagenet',input_shape=shape)
    # Freeze the base_model
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def MobileNetV2model(no_classes,shape):
    # Download the architecture with ImageNet weights
    base_model = MobileNetV2(include_top=False,input_shape=shape)
    # Freeze the base_model
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

if args["model"] == 'MobileNet':
    model = MobileNetmodel(8,shape=(224, 224,3))
    model.load_weights(os.path.join(path,'models/MobileNet_weights.h5'))
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
else:
    model = MobileNetV2model(8,shape=(224, 224,3))
    model.load_weights(os.path.join(path,'models/MobileNetV2_weights.h5'))
    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

print(" * Model and weights loaded!")

img_width, img_height = 224, 224
img = image.load_img(args['image'], target_size = (img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
predictions = model.predict(img,batch_size=1,verbose=1).tolist()
print("* Prediction")
print('The provided sample is most likely', np.argmax(predictions, axis = 1))
print('benign = 1','cerber = 2','crowti = 3','gandcrab = 4','genasom = 5','locky = 6 ','tescrypt = 7','wannacrypt = 8')
