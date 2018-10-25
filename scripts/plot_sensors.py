import json
import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
sys.path.append('/dycog/Jeremie/Loic/v2')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.db import ObjectSaverJoblib as ObjectSaver

from datasets.rsvp import RSVP_COLOR_116
from datasets.meeg import P300_SPELLER_MEEG

import matplotlib.pyplot as plt

rsvp = RSVP_COLOR_116()
meeg = P300_SPELLER_MEEG()

rsvp_raw_array = rsvp.get_subject_data(rsvp.subjects()[0], calib=True)
meeg_raw_array = meeg.get_subject_data(meeg.subjects()[0], calib=True)

rsvp_raw_array.ch_names
exclude_channels = ['P9', 'P10', 'PO9', 'PO3', 'PO4', 'PO10', 'Oz', 'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FT7', 'FT8']

def get_group_index(raw, exclude_channels):
    included_index = []
    exluded_index = []
    for index, ch in enumerate(raw.ch_names):
        if ch in exclude_channels:
            exluded_index.append(index)
        else:
            included_index.append(index)
    return [[], included_index, exluded_index]

rsvp_raw_array.plot_sensors(show_names=True, ch_type='eeg', ch_groups=get_group_index(rsvp_raw_array, exclude_channels), title='RSVP sensors')
meeg_raw_array.plot_sensors(show_names=True, ch_groups=get_group_index(meeg_raw_array, exclude_channels), title='P300 sensors', bgcolor='black')
plt.show()
