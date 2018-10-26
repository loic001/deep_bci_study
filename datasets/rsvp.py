import os
import shelve
import pathlib
import numpy as np
import mne
import scipy.io as sio
from scipy.io import loadmat

from datasets.base import EEGDataset

DATA_DIR = '/media/lolo/Maxtor/deep_bci_study/data/RSVP/'

class RSVP_COLOR_116(EEGDataset):
    # EEGDataset implementation
    def get_name(self):
        return 'RSVP_COLOR_116'

    def subjects(self):
        # VPgce
        return ['VPgcc', 'VPfat', 'VPgcb', 'VPgcg', 'VPgcd', 'VPgcf', 'VPgch', 'VPiay', 'VPicn', 'VPicr', 'VPpia']

    def get_subject_data(self, subject, calib=False):
        data_folder = DATA_DIR
        filename = os.path.join(data_folder, 'RSVP_Color116ms{}.mat'.format(subject))
        mne_raw_array = self._build_mne_raw_array_from_filename(filename, calib)
        return mne_raw_array

    def _extract_basic_data(self, data):
        set_1 = [x[0] for x in data['nfo']['clab'][0][0][0]]

        # electrode used in the experiment
        set_2 = [x[0] for x in data['bbci']['bbci'][0][0]['analyze'][0][0]['features'][0][0]['clab'][0][0][0]]

        all_relevant_channels = []
        channels_names = []
        for i in range(len(set_1)):
            if (set_1[i] in set_2):
                all_relevant_channels.append(data["ch%d" % (i + 1)].flatten())
                channels_names.append(set_1[i])
        marker_positions = data['mrk']['pos'][0][0][0]
        target = data['mrk']['y'][0][0][0]
        freq = data['mrk']['classified'][0][0][0][0][2]
        return all_relevant_channels, channels_names, marker_positions, target, freq

    def _extract_spelling_sequence_data(self, data):
        train_trial = data['mrk']['trial_idx'][0][0][0]
        train_block = data['mrk']['block_idx'][0][0][0]
        stimulus = data['mrk']['stimulus'][0][0][0]
        stimulus_names = [item[0] for item in data['mrk']['classified'][0][0][0][0][4][0]]

        train_mode = np.zeros(data['mrk']['mode'][0][0].shape[1]).astype(np.int8)

        train_mode[np.where(data['mrk']['mode'][0][0][0] == 1)[0]] = 1
        train_mode[np.where(data['mrk']['mode'][0][0][1] == 1)[0]] = 2
        train_mode[np.where(data['mrk']['mode'][0][0][2] == 1)[0]] = 3

        return stimulus, train_trial, train_block, train_mode, stimulus_names
    # private functions
    def _build_mne_raw_array_from_filename(self, filename, calib):
        name = filename.rstrip(os.path.sep).split('/')[-1] + ('_calib' if calib else '')
        data = sio.loadmat(filename)

        all_channels, ch_names, marker_positions, events, freq = self._extract_basic_data(data)
        stimulus, train_trial, train_block, train_mode, stimulus_names = self._extract_spelling_sequence_data(data)

        marker_positions = marker_positions[train_mode == 1] if calib else marker_positions[train_mode != 1]
        stimulus = stimulus[train_mode == 1] if calib else stimulus[train_mode != 1]
        events = events[train_mode == 1] if calib else events[train_mode != 1]

        cnt = np.asarray(all_channels)
        sfreq = freq
        ch_types = ["eeg" for _ in range(len(ch_names))]
        montage = mne.channels.read_montage("standard_1020")
        info = mne.create_info(
            ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, montage=montage)
        raw = mne.io.RawArray(cnt, info, verbose='WARNING')
        raw.info['ch_pos'] = list(zip(montage.ch_names, tuple(map(tuple, montage.get_pos2d()))))
        raw.info['name'] = name
        raw.info['y'] = np.column_stack((marker_positions, np.zeros(len(marker_positions), dtype='int'), events))
        raw.info['y_names'] = {0: 'Non-Target', 1: 'Target'}
        raw.info['y_stim'] = stimulus
        raw.info['y_stim_names'] = stimulus_names
        raw.info['vocab_length'] = len(raw.info['y_stim_names'])

        #rsvp -> repetition_length == vocab_length
        raw.info['repetition_length'] = raw.info['vocab_length']

        nb_repetitions = 10
        raw.info['nb_items'] = len(raw.info['y_stim']) // raw.info['repetition_length'] // nb_repetitions
        raw.info['nb_repetitions_by_item'] = np.full((raw.info['nb_items'],), nb_repetitions)
        print(raw.info)

        return raw
