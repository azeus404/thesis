# Reade Me 
Updated 09-10-2020

# Aim and objectives
We aim to achieve a high accuracy and low false postive classier of Windows ransomware files. 
By using minimal samples (low resources) and deep neural Networks.

# How can Deep Learning be used to detect & identify ransomware for the Microsoft Windows operating system?

To answer the main central research question, two theoretical and an empirical sub-question have been formulated:

1.	What are the current methods for detecting & identifying ransomware?
2.	What is Deep Learning & Deep Transfer Learning?
3.	How can a pre-trained model be implemented for detecting & identifying ransomware?

## The proposed method consists of the following basic steps: 
1) Given a dataset of labeled ransomware executables, convert each sample to its respective byteplot grayscale image; 
2) Convert each byteplot to an RGB image, rescaling it to 224 × 224 dimension and subtracting the mean RGB value computed on the ImageNet dataset from each pixel, as proposed by Krizhevsky et al. [9], to feed it to the deep neural network; 
3) Build a deep convolutional neural network (DCNN) based on the different architectures, 
replacing the last 1000 fully-connected softmax layer by a 8 fully-connected softmax layer and transferring the parameters of architecture convolutional layers to the convolutional layers of the DCNN models;
4) Freeze the transferred convolutional layer’s parameters and train the DCNN model to classify each sample into its ransomware family. An overview of the entire method’s pipeline is given in figure x.


# Setup 
We use diffrent virtual environments (pipevn) to import and configure the necessary software libaries.
Please consult https://pypi.org/project/pipenv/ how to use.

# Parsing and prepocessing
Before using the proposed_approach.ipynb in the notebooks folder (used in chapter 4 of the thesis) use the (pre-)processing scripts in the preprocessing folder.
This is a list of all scripts used in oder of use:
1 - parse.py
2 - binar2image.py (to iterm folder)
3 - imagesize.py 
4 - preprocess.py distribute
5 - process.py generates trainset.plk and testset.plk

To collect the samples from the a Windows 10 Operating system use the script as provided in the windows folder.

## Expertiment
The experiment can be re runed by using the files in de compare-dnn folder
reset the folderpaths in the scripts to match the new environment and execut the runner.sh script 
The evaluate_models.ipynb notebook in notebooks folder is used to evualuate the models performance during training, validation and testing.

## Experimental results
Use the expertimental_results.ipynb (Chapter 5 of the thesis) in the notebooks folder.

