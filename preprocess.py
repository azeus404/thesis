import json
import argparse
import os
import pandas as pd
import re
import shutil
import numpy as np
from threading import Thread
from sklearn.model_selection import StratifiedShuffleSplit
import logging

"""
Updated
"""

logfile = 'preprocesslog-selection.log'

logging.basicConfig(filename=logfile,level=logging.INFO, format="%(asctime)s - %(message)s")

parser = argparse.ArgumentParser(description='Functions to process interim data')

parser.add_argument('action',
                       metavar='action',
                       type=str,
                       help='action')

args = parser.parse_args()

def distibute_ransomware_samples(data, interim_path="/home/labuser/deeplearning/thesis/datasets/interim",processed_path="/home/labuser/deeplearning/thesis/datasets/preprocessed/gray",datasetname="vtdataset"):
    """
    Copy ransomware image samples from interim to preprocessed folder
    """

    preselection = data[(data ['imagesize_w'] >= 112) & (data['imagesize_h'] >= 112)]
    selection = preselection.groupby('label').filter(lambda x: len(x) >= 100) # select label greater equal to 100 samples

    for path ,label in zip(selection['path'], selection['label']):
        src = path.replace('.json','.png')

        print('[*] copy ',src)
        file = os.path.split(src)[1]
        labeldir = os.path.join(processed_path,label)
        dst = os.path.join(labeldir,file)
        print('[*] to',dst)
        if not os.path.exists(labeldir):
            os.makedirs(labeldir)
        try:
            Thread(target=shutil.copy, args=[src, dst]).start()
            logging.info('Copied %s to %s', str(src), str(dst))
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

def main(input_action):
    if input_action == 'distribute':
        print("[*] Copy sample files to processed location")
        df = pd.read_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/images_size_grayscale.pkl')
        distibute_ransomware_samples(df)
    else:
        exit()
if __name__ == "__main__":
    if os.path.isfile(logfile):
        os.remove(logfile)
    input_action = args.action
    main(input_action)
