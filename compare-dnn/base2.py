#!/bin/env python3
import sys
import os
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from datetime import datetime
import shutil
import math
import itertools
import numpy as np
from datetime import datetime
import argparse
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger,LearningRateScheduler
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

from keras import Model, Input
from keras.optimizers import Adam,SGD

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Grayscale fix https://stackoverflow.com/questions/42462431/oserror-broken-data-stream-when-reading-image-file
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#local
from config import config as config
from helpers import helpers
from models import models as customm

#Implemented Models
MODELS = [
	'ResNet50',
	'ResNet50V2',
	'ResNet101V2',
	'ResNet152V2',
	'InceptionV3',
	'Xception',
	'VGG16',
	'VGG19',
	'MobileNet',
	'MobileNetV2',
	'EfficientNetB7',
	'InceptionResNetV2',
	'DenseNet121',
	'DenseNet169',
	'NASNetMobile',
	'NASNetLarge',
	'Custom'
]

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, type=str, default="VGG16",
	help="name of pre-trained network to use")
ap.add_argument("-l", "--layers", required=False,type=float, default="1.0",
	help="unfreeze 0% of layers default: 100")
args = vars(ap.parse_args())

if args["model"] not in MODELS:
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

modelname = args["model"]

print("[****] CNN %s - model ransomware VTsamples imagenet weights 2020 [****]" % modelname)
"""
    https://github.com/azeus404/thesis
	Beta __version__
	Updated 30-09-2020
	added fixed inputshapes
	added NASNetLarge
	https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/
"""
print('[*] Cleaning up previous runs')
run_time = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(os.path.join('results',modelname)):
	os.mkdir(os.path.join('results',modelname))
else:
	shutil.rmtree(os.path.join('results',modelname))
	os.mkdir(os.path.join('results',modelname))


result_path = os.path.join('results',modelname)

print('[*] Tensorflow version = ', (tf.__version__))
print('[*] Keras version = ', tf.keras.__version__)

plt.style.use('seaborn')
sns.set(font_scale=2)

print('[*] Activating GPU')
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

print("[*] Loading Model configuration")
batch_size = 64 #64 #32 #265 #128 #config.batch_size
validation_size = 0.2 #config.validation_size
img_width, img_height, img_num_channels = 112, 112,3 #112, 112,3 # 37,632 features
image_size = (img_width, img_height)
loss_function = config.loss_function
no_epochs = config.no_epochs
no_classes = config.no_classes
seed = 12
optimizer = config.optimizer
lr = config.lr
verbosity = config.verbosity

print('[*] Loading dataset paths')

testimg_path = config.testimg_path

#Grayscale images
#training_path = '/home/labuser/deeplearning/thesis/datasets/processed/GRAYSCALE/train'
#testing_path = '/home/labuser/deeplearning/thesis/datasets/processed/GRAYSCALE/test'

#RGB images
training_path = config.training_path
testing_path = config.testing_path



print("[*] Pre-trained model - %s loaded" % modelname)
"""

"""
if modelname == 'ResNet50':
	model = customm.ResNet50model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'ResNet50V2':
	model = customm.ResNet50V2model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'ResNet101V2':
	model = customm.ResNet101V2model(no_classes=config.no_classes,shape=(224,224, img_num_channels))
elif modelname == 'ResNet152V2':
	model = customm.ResNet152V2model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'InceptionV3':
	model = customm.InceptionV3model(no_classes=config.no_classes,shape=(299, 299,img_num_channels))
elif modelname == 'Xception':
	model = customm.Xceptionmodel(no_classes=config.no_classes,shape=(299, 299, 3))
elif modelname == 'VGG19':
	model = customm.VGG19model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'VGG16':
	model = customm.VGG16model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'MobileNet':
	model = customm.MobileNetmodel(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'MobileNetV2':
	model = customm.MobileNetV2model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'EfficientNetB0':
	model = customm.EfficientNetB0model(no_classes=config.no_classes,shape=(img_width, img_height, img_num_channels))
elif modelname == 'EfficientNetB1':
	model = customm.EfficientNetB1model(no_classes=config.no_classes,shape=(img_width, img_height, img_num_channels))
elif modelname == 'EfficientNetB2':
	model = customm.EfficientNetB2model(no_classes=config.no_classes,shape=(img_width, img_height, img_num_channels))
elif modelname == 'EfficientNetB3':
	model = customm.EfficientNetB3model(no_classes=config.no_classes,shape=(img_width, img_height, img_num_channels))
elif modelname == 'EfficientNetB7':
	model = customm.EfficientNetB7model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'InceptionResNetV2':
	model = customm.InceptionResNetmodel(no_classes=config.no_classes,shape=(299, 299, img_num_channels))
elif modelname == 'DenseNet121':
	model = customm.DenseNet121model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'DenseNet169':
	model = customm.DenseNet169model(no_classes=config.no_classes,shape=(224, 224, img_num_channels))
elif modelname == 'NASNetMobile':
	model = customm.NASNetMobilemodel(no_classes=config.no_classes,shape=(224, 224,img_num_channels))
elif modelname == 'NASNetLarge':
	model = customm.NASNetLargemodel(no_classes=config.no_classes,shape=(331, 331,img_num_channels))
elif modelname == 'Custom':
	model = customm.custom_malware_model(no_classes=config.no_classes,shape=(64,64,3))
	image_size = (64, 64)


print("[*] Spliting data - {training}% training - {validation}% validation - {testing}% testing".format(training=(100-(validation_size*100)),validation=(validation_size*100),testing=10.0))
#img = Image.open('lena.png').convert('LA')
#train_image_generator = ImageDataGenerator(validation_split=validation_size, rescale=1.0 / 255)
train_image_generator = ImageDataGenerator(validation_split=validation_size,samplewise_center=True, samplewise_std_normalization=True)

test_image_generator = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
#test_image_generator = ImageDataGenerator(rescale=1.0 / 255)

train_set = train_image_generator.flow_from_directory(training_path,
                                                 target_size = (img_width, img_height),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical',
												 shuffle = True,
												 subset = 'training',
                                                 seed = seed
												 )
val_set = train_image_generator.flow_from_directory(training_path,
                                                 target_size = (img_width, img_height),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical',
												 shuffle = True,
												 subset = 'validation',
                                                 seed = seed
												 )
test_set = test_image_generator.flow_from_directory(directory = testing_path,
													target_size = (img_width, img_height),
													batch_size = 1,
													class_mode ='categorical',
													shuffle = False
													)


print("[*] Save plot of baseline model to file")
plot_model(model, to_file= os.path.join(result_path, modelname + '_model.png'), show_shapes=True)

print("[*] Describe baseline model")
print(model.summary())

print("[*] Dealing with imbalanced dataset - calculating class weights")
class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_set.classes),
                train_set.classes)
class_weights = dict(enumerate(class_weights))


#print("[*] Learning rate scheduler")
def scheduler(epoch, lr):
	"""
	Decrease learning rate after 10 epochs
	: epoch
	: lr
	"""
	if epoch < 10:
		return lr
	else:
		return lr * tf.math.exp(-0.1)

print("[*] Training the top layer of the model - using pre-preprocessed image size %s " % str(image_size))
top_weights_path = os.path.join(result_path, modelname + '_top_model_weights.h5')
log = os.path.join(result_path,modelname +'_training.log')

if os.path.exists(top_weights_path):
	os.remove(top_weights_path)

startTime = datetime.now()

callbacks = [
    ModelCheckpoint(top_weights_path, save_best_only=True, monitor='val_loss',mode='auto'),
	EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto',restore_best_weights=True),
	#LearningRateScheduler(scheduler),
	tf.keras.callbacks.TerminateOnNaN(),
    CSVLogger(log),
]

model.compile(
	#High learning rate default=0.001 [1e-3] = 1 x 10 -3
    optimizer= Adam(learning_rate=1e-3),
    loss= loss_function,
    metrics=["accuracy"],
)

history = model.fit(
    train_set, epochs=no_epochs, callbacks=[callbacks], steps_per_epoch=math.ceil(train_set.samples // batch_size), validation_steps=math.ceil(val_set.samples // batch_size), validation_data=val_set, class_weight=class_weights, verbose=verbosity
)
print(round(model.optimizer.lr.numpy(), 5))
print("[*] Save weights %s"% top_weights_path)
model.save_weights(top_weights_path)
model.load_weights(top_weights_path)
print("[+] Total pre-training time {0} ".format(datetime.now() - startTime))

print("[*] Generate diagnostics plots ")
helpers.summarize_diagnostics(os.path.join(result_path,modelname), history)
helpers.plot_history(os.path.join(result_path,modelname), history)

print("[*] Evaluating on validation data")
val_loss,val_acc = model.evaluate(val_set, batch_size=batch_size)
print("[*] Validation accuracy: {:.2f}%".format(100*val_acc))
print("[*] Validation loss: {:.2f}".format(val_loss))
print("[+] Total training time", datetime.now() - startTime)

print("[*] Evaluating on testdata")
test_score = model.evaluate(test_set, verbose=2)
data = {'test loss':[test_score[0]] ,'test acc':[test_score[1]]}
df = pd.DataFrame.from_dict(data)
df.to_pickle(os.path.join(result_path,modelname + "_testresults.pkl"))

print("[*] Test accuracy: {:.2f}%".format(test_score[1] * 100))
print("[*] Test loss: {:.2f}".format(test_score[0]))

print("[*] Saving confusion Matrix - on test data")
target_names = []
for key in train_set.class_indices:
    target_names.append(key)

Y_pred = model.predict(x=test_set,steps=len(test_set),verbose=0)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(test_set.classes, y_pred)
helpers.plot_confusion_matrix(os.path.join(result_path,modelname),cm, target_names, title='Confusion Matrix - %s' % modelname)

print('[*] Saving classification Report')
report = classification_report(test_set.classes, y_pred, target_names=target_names,output_dict=True)

from sklearn.metrics import precision_score,recall_score,f1_score
print('precision {:.2f} macro'.format(precision_score(test_set.classes, y_pred, average='macro')))
print('recall {:.2f} macro'.format(recall_score(test_set.classes, y_pred, average='macro')))
print('f1 score {:.2f} macro'.format(f1_score(test_set.classes, y_pred, average='macro')))
df = pd.DataFrame.from_dict(
	{'model':modelname,
	'f1_score':[f1_score(test_set.classes, y_pred, average='macro')],
	'recall_score':[recall_score(test_set.classes, y_pred, average='macro')],
	'precision_score':[precision_score(test_set.classes, y_pred, average='macro')],
	'test_loss':[test_score[0]],
	'test_acc':[test_score[1]]
	})
df.to_pickle(os.path.join(result_path,modelname + "_scores.pkl"))

df = pd.DataFrame.from_dict(report)
df.to_pickle(os.path.join(result_path,modelname + "_classification_report.pkl"))
print(report)
print("[*] Saving model weights")
model.save_weights(os.path.join(result_path,modelname + "_weights.h5"))

print('[-] Clearing memory')
tf.keras.backend.clear_session()
