import os
import json
import logging
import re
import pandas as pd
import argparse

logging.basicConfig(filename='parselog.log',level=logging.INFO, format="%(asctime)s - %(message)s")

parser = argparse.ArgumentParser(description='Processing VTdataset')

parser.add_argument('masterdata',
                       metavar='path',
                       type=str,
                       help='Location of masterdata')
args = parser.parse_args()

def cleanup_result(value):
    """
        Fix typos in labels made by Microsoft
    """
    if re.search('grandcrab',value.lower()):
        return 'gandcrab'
    elif re.search('firecerb',value.lower()):
        return 'cerber'
    elif re.search('critroni',value.lower()):
        return 'citroni'
    else:
        return value.lower()


def fileList(path,datasetname='vtdataset'):
    """
    Processing masterdata folders select only if json file
    and executable exist and valid json file
    """
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print("[*] Collecting sample {0} - {1}".format(filename,filename))
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as jfile:
                    try:
                        result = json.load(jfile)
                        matches.append({'path': filepath})
                    except Exception as e:
                        logging.error('Exception occurred on file %s ' % filepath)
                        pass
    return pd.DataFrame.from_dict(matches)

def process_sample_selection(data,datasetname='vtdataset'):
    """
    First selection criteria:
    Sample selection based on Microsoft classifcation and use it at gound truth label
    """
    dict = []
    for item in data['path']:

        with open(item, 'r') as jfile:

            result = json.load(jfile)
            first_seen = result['first_seen']

            for key, value in result['scans'].items():
                if value['detected'] == True and key in ['Microsoft']:
                    print('[*] Match found {0} - {1}'.format(str(key),str(value['result'])))
                    result = str(value['result'])
                    filehash = os.path.split(item)[1]
                    fsize = os.stat(item).st_size
                    filesizeKB = round(fsize / 1024, 3)
                    print(filehash)
                    if re.search('\.', result):
                        clean = str(value['result']).split('.')[0]
                    elif re.search('\!', result):
                        clean = str(value['result']).split('!')[0]
                    else:
                        clean = str(value['result'])
                    dict.append({'scanner':str(key),'path':str(item),'filehash':filehash.replace('.json',''),'result':cleanup_result(str(value['result'])),'label': clean.lower(),'filesizeKB':filesizeKB,'first_seen':str(first_seen)})
    return pd.DataFrame.from_dict(dict)

def dedup_data_cleanup(df,datasetname="vtdataset"):
        """
            Data cleansing 1: remove duplicate filehashes
        """
        df.sort_values(['filehash'], inplace = True)
        print('Duplicates found {0}'.format(df[df.duplicated('filehash')].count()))
        df.drop_duplicates(subset ='filehash',keep = False, inplace = True)
        return df

def main(input_path):
    df = fileList(input_path)
    df.to_pickle('/home/labuser/deeplearning/thesis/solutions/dataman/parsed.pkl')

    #df = process_sample_selection(df)
    #df.to_pickle('/home/labuser/deeplearning/thesis/solutions/dataman/preprocessed.pkl')

    #df= dedup_data_cleanup(df)
    #df.to_pickle('/home/labuser/deeplearning/thesis/solutions/dataman/preprocessed_1.pkl')

if __name__ == "__main__":
    if os.path.isfile('parselog.log'):
        os.remove('parselog.log')
    input_path = args.masterdata
    if not os.path.isdir(input_path):
        print('The path specified does not exist')
        sys.exit()
    main(input_path)
