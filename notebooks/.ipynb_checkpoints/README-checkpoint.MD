INSPIRED BY https://lasca.ic.unicamp.br/paulo/papers/2017-ICMLA-edmar.rezende-malicious.classif.transfer.learning.ResNet-50.pdf
# Aim and objectives
We aim to achieve a high accuracy and low false postive classier of Windows ransomware files. 
By using minimal samples (low resources) and deep neural Networks.

# How can Deep Learning be used to detect & identify ransomware for the Microsoft Windows platform?

To answer the main central research question, two theoretical and an empirical sub-question have been formulated:

1.	What are the current methods for detecting & identifying ransomware?
2.	What is Deep Learning & Deep Transfer Learning?
3.	How can a pre-trained model be implemented for detecting & identifying ransomware?


## The proposed method consists of the following basic steps: 
1) Given a dataset of labeled ransomware executables, convert each sample to its respective byteplot grayscale image; 
2) Convert each byteplot to an RGB image, rescaling it to 224 × 224 dimension and subtracting the mean RGB value computed on the ImageNet dataset from each pixel, as proposed by Krizhevsky et al. [9], to feed it to the deep neural network; 
3) Build a deep convolutional neural network (DCNN) based on the 6 architectures, 
replacing the last 1000 fully-connected softmax layer by a 8 fully-connected softmax layer and transferring the parameters of architecture convolutional layers to the convolutional layers of the DCNN models;
4) Freeze the transferred convolutional layer’s parameters and train the DCNN model to classify each sample into its ransomware family. An overview of the entire method’s pipeline is given in figure x.


# Method
In our method, we use MobileNet as the base model, pretrained for object detection task on the ImageNet dataset [10].
Our hypothesis is that despite the disparity between natural images and malware byteplot images, MobileNet comprehensively trained on the ImageNet may still be transferred to
make ransomware image recognition tasks more effective, since collecting and annotating large numbers of samples still poses significant challenges.

## Models

Traditional models:
- VGG16
- VGG19
- ResNet

Modern models:
- InceptionV3
- Xception
- MobileNetV2
https://machinethink.net/blog/mobilenet-v2/

# Conclusions

