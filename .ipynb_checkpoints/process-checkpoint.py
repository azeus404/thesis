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
Updated clean to label 19-08-2020
"""

logfile = 'processlog.log'

logging.basicConfig(filename=logfile,level=logging.INFO, format="%(asctime)s - %(message)s")



parser = argparse.ArgumentParser(description='Functions to process')

parser.add_argument('action',
                       metavar='action',
                       type=str,
                       help='action')

args = parser.parse_args()

def select_stratified_test_samples(data,datasetname="vtdataset"):
    """
        Select a 10% stratified test samples
        source: https://medium.com/@411.codebrain/train-test-split-vs-stratifiedshufflesplit-374c3dbdcc36
    """
    preselection = data[(data ['imagesize_w'] >= 112) & (data['imagesize_h'] >= 112)]
    selection = preselection.groupby('label').filter(lambda x: len(x) >= 100)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index , test_index in split.split(selection, selection['label']):
        strat_train_set = selection.iloc[train_index]
        strat_test_set = selection.iloc[test_index]
    return (strat_test_set.to_pickle("./testset.pkl"), strat_train_set.to_pickle("./trainset.pkl"))


def distibute_traindata(data, processed_path="/home/labuser/deeplearning/thesis/datasets/preprocessed",train_path="/home/labuser/deeplearning/thesis/datasets/processed/train"):
        """
        Copy ransomware image samples from preprocessed to processed folder
        """
        for path ,label in zip(data['path'], data['label']):
            src = path.replace('.json','.png')

            print('[*] copy ',src)
            file = os.path.split(src)[1]
            labeldir = os.path.join(train_path,label)
            dst = os.path.join(labeldir,file)
            print('[*] to',dst)
            if not os.path.exists(labeldir):
                os.makedirs(labeldir)
            try:
                Thread(target=shutil.copy, args=[src, dst]).start()
                logging.info('Copied %s to %s', str(src), str(dst))
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)


def distibute_testdata(data, processed_path="/home/labuser/deeplearning/thesis/datasets/preprocessed",test_path="/home/labuser/deeplearning/thesis/datasets/processed/test"):
        """
        Copy ransomware image samples from preprocessed to processed folder
        """
        for path ,label in zip(data['path'], data['label']):
            src = path.replace('.json','.png')

            print('[*] copy ',src)
            file = os.path.split(src)[1]
            labeldir = os.path.join(test_path,label)
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
        print("[*] Copy sample files to preprocessed location")
        df = pd.read_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/images_size.pkl')
        #select_stratified_test_samples(df)

        print("[*] Copy sample files to processed location - train ")
        df = pd.read_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/trainset.pkl')
        distibute_traindata(df)

        print("[*] Copy sample files to processed location - test ")
        df = pd.read_pickle('/home/labuser/deeplearning/thesis/datamanagement/dataman/testset.pkl')
        distibute_testdata(df)


    else:
        exit()

if __name__ == "__main__":
    if os.path.isfile(logfile):
        os.remove(logfile)
    input_action = args.action
    main(input_action)
