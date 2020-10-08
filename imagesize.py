import os
from PIL import Image
import pandas as pd


def image_size(path):
    im = Image.open(path)
    return im.size

def fileList(source):
    """
        find images in preprocessd folder and report on image size and file size in bytes
    """
    matches = []
    for root, dirnames, filenames in os.walk(source):
        for filename in filenames:
            print("[*] Collecting images %s "% filename)
            if filename.endswith(".png"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r") as jfile:
                    try:
                        w,h = image_size(filepath)
                        label = filepath.split('/')[-2]
                        print(label)
                        filesize = os.stat(filepath).st_size
                        matches.append({"path": filepath,'label': label , "imagesize_w":w,'imagesize_h':h,'filesize':filesize})
                    except Exception as e:
                        logging.error("Exception occurred on file %s " % filepath)
                        pass
    return matches

data = fileList('/home/labuser/deeplearning/thesis/datasets/interim')
df = pd.DataFrame.from_dict(data)
#df.to_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/images_size.pkl')
#df.to_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/images_size_grayscale.pkl')
