import json
import os
import sys
import argparse
import logging
import gc
logging.basicConfig(level=logging.DEBUG)

sys.path.append('/dycog/Jeremie/Loic/v2')
from datasets.base import P300Dataset
from utils.db import ObjectSaverJoblib as ObjectSaver

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", help='output dir', type=str, required=True)
args = parser.parse_args()
dir = args.input_dir

memmap_dir = os.path.join(dir, 'memmap')
if not os.path.exists(memmap_dir):
    os.makedirs(memmap_dir)
with open(os.path.join(dir, 'data_dict.json'), encoding='utf-8') as f:
    data_dict = json.loads(f.read())

saver = ObjectSaver(dir)

for subject, datasets in data_dict.items():
    print(subject)
    calib_dataset_key = datasets.get('calib_dataset')
    print(calib_dataset_key)
    calib_dataset = saver.load(calib_dataset_key)
    P300Dataset.to_memmap(calib_dataset, memmap_dir, subject+'_calib')
    calib_dataset = None
    dataset_key = datasets.get('dataset')
    dataset = saver.load(dataset_key)
    P300Dataset.to_memmap(dataset, memmap_dir, subject)
    dataset = None
    gc.collect()
