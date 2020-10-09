#!/bin/env python3
import tensorflow as tf
from keras import Model, Input, Sequential

from keras.layers import Conv2D,GlobalAveragePooling2D,Convolution2D,MaxPool2D
from keras.layers import Dropout, Activation, Dense, Flatten,MaxPooling2D,BatchNormalization

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet_v2 import ResNet50V2,ResNet101V2,ResNet152V2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import NASNetLarge
#keras.applications.resnet.preprocess_input
#keras.applications.resnet_v2.preprocess_input
#keras.applications.inception_v3.preprocess_input
#keras.applications.xception.preprocess_input
#keras.applications.vgg16.preprocess_input
#keras.applications.vgg19.preprocess_input
#keras.applications.mobilenet_v2.preprocess_input

"""
    Typical tranfser scenario as described on https://keras.io/guides/transfer_learning
"""

def ResNet50V2modelA(no_classes,shape):
    """
        Deep CNN 50 layers
    """
    base_model = ResNet50V2(include_top=False, weights='imagenet',input_shape=shape)
    # Taking the output of the last convolution block in ResNet50
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax')(x)
    model = Model(base_model.input, outputs=predictions)
    # Training only top layers i.e. the layers which we have added in the end
    for layer in base_model.layers:
        layer.trainable = False
    return model

def ResNet50V2model(no_classes,shape):
    """
        Deep CNN 50 layers
    """
    base_model = ResNet50V2(include_top=False, weights='imagenet',input_shape=shape)
    # Freeze the base_model
    base_model.trainable = False
    inputs = Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu',name='predictions')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(no_classes, activation= 'softmax')(x)
    model = Model(inputs, outputs=predictions)
    return model

def ResNet101V2model(no_classes,shape):
    """
        Deep CNN 50 layers
    """
    base_model = ResNet101V2(include_top=False, weights='imagenet',input_shape=shape)
    # Freeze the base_model
    base_model.trainable = False
    inputs = Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu',name='predictions')(x)
    predictions = Dense(no_classes, activation= 'softmax')(x)
    model = Model(inputs, outputs=predictions)
    return model

def ResNet152V2model(no_classes,shape):
    """
        Deep CNN 50 layers
    """
    base_model = ResNet152V2(include_top=False, weights='imagenet',input_shape=shape)
    # Freeze the base_model
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu',name='predictions')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(no_classes, activation= 'softmax')(x)
    model = Model(inputs, outputs=predictions)
    return model



def ResNet50model(no_classes,shape):
    """
        Deep CNN 50 layers
        Example based on https://keras.io/guides/transfer_learning/
    """
    base_model = ResNet50(include_top=False, weights='imagenet',input_shape=shape)
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

def InceptionV3model(no_classes,shape):
    base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def Xceptionmodel(no_classes,shape):
    base_model = Xception(include_top=False,weights='imagenet',input_shape=shape )
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    #x = Dense(2048, activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model


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
    base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=shape)
    # Freeze the base_model
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def VGG16model(no_classes,shape):
    # Download the architecture of VGG16 with ImageNet weights
    base_model = VGG16(include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    #x = Dense(4096, activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def VGG19model(no_classes,shape):
    # Download the architecture of VGG19 with ImageNet weights
    base_model = VGG19(include_top=False, weights='imagenet',input_shape=shape)
    #base_model = VGG19(include_top=True, weights='imagenet',in)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.2)(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def InceptionResNetmodel(no_classes,shape):
    base_model = InceptionResNetV2(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def EfficientNetB0model(no_classes,shape):
    """
    EfficientNetB0
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    Uses a fixed input size 224,224
    """
    base_model = EfficientNetB0(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def EfficientNetB1model(no_classes,shape):
    """
    EfficientNetB1
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    Uses a fixed input size 224,224
    """
    base_model = EfficientNetB1(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def EfficientNetB2model(no_classes,shape):
    """
    EfficientNetB2
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    Uses a fixed input size 224,224
    """
    base_model = EfficientNetB2(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def EfficientNetB3model(no_classes,shape):
    """
    EfficientNetB2
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    Uses a fixed input size 224,224
    """
    base_model = EfficientNetB3(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def EfficientNetB7model(no_classes,shape):
    """
    EfficientNetB7
    https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
    Uses a fixed input size 224,224
    """
    base_model = EfficientNetB7(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def DenseNet121model(no_classes,shape):
    """
    DenseNet121
    """
    base_model = DenseNet121(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def DenseNet169model(no_classes,shape):
    """
    DenseNet169
    """
    base_model = DenseNet169(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def NASNetMobilemodel(no_classes,shape):
    """
    NASNetMobile Learning Transferable Architectures for Scalable Image Recognition,2018
    """
    base_model = NASNetMobile(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    #x = Dense(1024,activation='relu')(x)
    x = Dense(1056, activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def NASNetLargemodel(no_classes,shape):
    """
    NASNetLarge Learning Transferable Architectures for Scalable Image Recognition,2018
    """
    base_model = NASNetLarge(include_top=False, weights='imagenet',input_shape=shape)
    base_model.trainable = False
    inputs = Input(shape=shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    #x = Dense(1024,activation='relu')(x)
    x = Dense(1056, activation='relu')(x)
    predictions = Dense(no_classes, activation= 'softmax',name='predictions')(x)
    model = Model(inputs, outputs=predictions)
    return model

def custom_malware_model(no_classes,shape):
    """
    NOT implemented!!
    Uses a fixed shape size of 64x64x1 = Grayscale
    https://towardsdatascience.com/malware-classification-using-convolutional-neural-networks-step-by-step-tutorial-a3e8d97122f
    """
    model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #             input_shape=(64,64,3)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(50, activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    return model
