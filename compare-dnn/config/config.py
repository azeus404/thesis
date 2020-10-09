#!/bin/env python3
batch_size = 64
validation_size = 0.1
img_width, img_height, img_num_channels = 112,112,3
image_size = (img_width, img_height)
loss_function = 'categorical_crossentropy'
no_classes = 8
no_epochs = 50
seed = 1337
optimizer = 'Adam'
lr = 0.0001
verbosity = 1

testimg_path = '/home/labuser/deeplearning/thesis/solutions/testdata/6ab3e38e2614144f72c70130f74395f9afba4c23c0c24d7aa85e942b25b59715_RGB.png'
training_path = '/home/labuser/deeplearning/thesis/datasets/processed/RGB/train'
testing_path = '/home/labuser/deeplearning/thesis/datasets/processed/RGB/test'
