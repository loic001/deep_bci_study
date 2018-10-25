import numpy as np
import mne
import os
import pathlib

#h5py lib
import deepdish as dd

from datasets.base import EEGDataset
from utils.mne import concat_raw_arrays_list

# DATA             :    EEG data (NSamples x Nchannels)
# EpochIsTarget          :    (NEpoch x 1 ) Flag to indicate if it's an evoked response to a flashed Target (1 : Target , 0 : Non Target)
# EventFlash               :    (NEpoch x 2 ) Event List => 1st column : n° sample of flash stimulation  2st column : Event code (not useful here)
# NbItemsTarget            :     Number of Items to select by P300 Speller
# ItemsTarget        :     List of Items to select by P300 Speller
# EventGoodFeedBack     :     (NbItemsTarget x 2)  Event List for the result feedback => 1st column : n° sample of feedback presentation  2st column : Good feedback flag (1 : Good feedback , 0 : Good feedback)
# ItemsFlashedPerEpoch:     (NEpoch x 6) Flashed Items per Epoch (ex :  8    12    13    20    30    34 =>  H L M T 4 8  are flashed in same time)
# ItemsOnScreen        :     (36 x 1 )  List of items presented on matrix
# NbChannels        :     Number of channels
# LabelChannels            :    (NbChannelsx3 char) Label of channels
# NbRepet4EachItemsTarget : (NbItemsTarget x 1 ) Nb repetition per item to select (ex : 2=> 2 x 12 flashs)
# SamplingFrequency    : Sampling frequency
# Subject            : Subject Id

DATA_DIR = '/deeplearning/data/P300_Speller_MEEG'

class P300_SPELLER_MEEG(EEGDataset):
    # EEGDataset implementation
    def get_name(self):
        return 'P300_SPELLER_MEEG'

    def subjects(self):
        #no S03, S12
        return ['S01', 'S02', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
                'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28']

    def get_subject_data(self, subject, calib=False):
        sessions_calib = ['01', '02']
        sessions = ['03', '04']

        data_folder = DATA_DIR
        _sessions = sessions_calib if calib else sessions

        raw_arrays = []
        for _session in _sessions:
            filename = os.path.join(data_folder, '{}_Sess{}.h5'.format(subject, _session))
            mne_raw_array = self._build_mne_raw_array(filename)
            raw_arrays.append(mne_raw_array)

        mne_raw_array = concat_raw_arrays_list(raw_arrays)
        if calib:
            mne_raw_array.info['name'] += '_calib'
        return mne_raw_array

    # private functions
    def _build_mne_raw_array(self, filename):
        name = filename.rstrip(os.path.sep).split('/')[-1]

        #data is a dict containing all the data from h5 file
        data = dd.io.load(filename)

        #ch_namesEpochIsTarget
        label_channels = data['LabelChannels']
        label_channels = [item.decode('UTF-8') for item in label_channels]
        ch_names = [label_channels[0][index] + label_channels[1][index] + label_channels[2][index] for index in range(len(label_channels[0]))]
        #patch 0 to O
        ch_names = [item.replace("0", "O").strip() for item in ch_names]


        items_on_screen = [item.decode('UTF-8') for item in data['ItemsOnScreen']]
        stimulus_names = {index+1:item for index, item in enumerate(items_on_screen)}

        # stimulus = data['EventFlash'].astype(int)[1]
        stimulus = data['ItemsFlashedPerEpoch'].astype(int).transpose()
        marker_positions = data['EventFlash'][0].astype(int)
        events = data['EpochIsTarget'][0].astype(int)
        nb_repetitions_by_item = data['NbRepet4EachItemsTarget'][0].astype(int)

        cnt = data['DATA']
        sfreq = data['SamplingFrequency'][0,0]
        ch_types = ["eeg" for _ in range(len(ch_names))]
        montage = mne.channels.read_montage("standard_1020")
        info = mne.create_info(
             ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage)
        raw = mne.io.RawArray(cnt, info, verbose='WARNING')
        raw.info['ch_pos'] = list(zip(montage.ch_names, tuple(map(tuple, montage.get_pos2d()))))
        raw.info['name'] = name
        raw.info['y'] = np.column_stack((marker_positions, np.zeros(len(marker_positions), dtype='int'), events))
        raw.info['y_names'] = {0: 'Non-Target', 1: 'Target'}
        raw.info['p300_speller_type'] = 'normal_splotch'
        raw.info['nb_repetitions_by_item'] = nb_repetitions_by_item
        raw.info['y_stim'] = stimulus
        raw.info['y_stim_names'] = stimulus_names
        raw.info['vocab_length'] = len(raw.info['y_stim_names'])
        raw.info['repetition_length'] = 12 #splotch
        raw.info['nb_items'] = data['NbItemsTarget'][0,0].astype(int)
        print(raw.info)
        return raw
