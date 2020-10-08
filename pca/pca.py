import os
import pandas as pd
from PIL import Image
import numpy as np
from numpy import asarray

def fileList(source):
    """
        find images in preprocessd folder and convert to numpyarray
        https://www.oreilly.com/library/view/programming-computer-vision/9781449341916/ch01.html
        Get 100 images and resize them to 112
    """
    i = 0
    samples = pd.DataFrame([])
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            print("[*] Collecting image %s "% filename)
            #if filename.endswith(".png") and i < 100: # 10 samples
            if filename.endswith(".png"):
                filepath = os.path.join(root, filename)
                file = Image.open(filepath).resize((112,112),Image.ANTIALIAS)
                img = asarray(file)
                #sample = img.reshape([3,-1]).T
                img = np.array(file).flatten() # due to flatting - no RGB but grayscale!!!
                sample = pd.Series(img,name=filepath )
                file.close()
                samples = samples.append(sample)
                i = i+1
    return samples.to_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/pca/prepeareforpca_noflat.pkl')
fileList('/home/labuser/deeplearning/thesis/datasets/preprocessed')
