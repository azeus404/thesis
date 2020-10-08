
import os
from os import listdir,makedirs
from os.path import isfile,join

from PIL import Image
#img = Image.open('image.png').convert('LA')
#img.save('greyscale.png')

path = r'/home/labuser/deeplearning/thesis/datasets/processed/test/crowti' # Source Folder
dstpath = r'/home/labuser/deeplearning/thesis/datasets/processed/gray/test/crowti' # Destination Folder

#try:
#    makedirs(dstpath)
#except:
#    print("Directory already exist, images will be written in asme folder")

# Folder won't used
files = [f for f in listdir(path) if isfile(join(path,f))]

for image in files:
    #try:
        print(os.path.join(path,image))
        #img = cv2.imread(os.path.join(path,image))
        img = Image.open(os.path.join(path,image)).convert('LA')
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dstPath = join(dstpath,image)
        img.save(dstPath)
        print(dstPath)
    #except:
    #    print("{} is not converted".format(image))
