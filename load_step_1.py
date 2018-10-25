import json
import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
sys.path.append('/dycog/Jeremie/Loic/v2')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.db import ObjectSaverJoblib as ObjectSaver

from datasets.rsvp import RSVP_COLOR_116
from datasets.meeg import P300_SPELLER_MEEG

from datasets.transformers import SamplingTransformer, ZScoreTransformer, InterpolatorTransformer
from datasets.loader import DatasetDefLoader

parser = argparse.ArgumentParser()
parser.add_argument("--cache-dir", help='cache dir', type=str, required=True)
parser.add_argument("--output-dir", help='output dir', type=str, required=True)

args = parser.parse_args()
cache_dir = args.cache_dir
saver_dir = args.output_dir

saver = ObjectSaver(saver_dir)
save_func = lambda obj, _id=None: saver.save(obj, _id)

datasets_def = [
    {
        'eeg_dataset': RSVP_COLOR_116(),
        'include_subjects': ['*'], #, 'VPfat', 'VPgcc', 'VPiay'
        'exclude_subjects': [],
        'tmin': 0.,
        'tmax': 0.6,
        'exclude_channels': ['P9', 'P10', 'PO9', 'PO3', 'PO4', 'PO10', 'Oz', 'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FT7', 'FT8'],
        #sfreq = 200 / decim_factor = 100
        'decim_factor' : 2,
        'apply_filter': {
            'l_freq': 1.,
            'h_freq': 20.
        },
        #epochs params
        # 'exclude_channels': ['P8', 'O2'],
        'transformers': [InterpolatorTransformer()],
        #others params
        'cache': True
    }
]

subjects_data_dict = DatasetDefLoader(cache_dir).loads(datasets_def, db_save=True, db_save_func=save_func)

file_path = os.path.join(saver_dir, 'data_dict.json')
text_file = open(file_path, "w")
text_file.write(json.dumps(subjects_data_dict, indent=4, sort_keys=True))
text_file.close()

print('finish')
