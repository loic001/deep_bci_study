import logging
import os
import numpy as np
import mne
import json
import sklearn
import imblearn
from collections import Counter
import gc
import datasets.transformers

import hashlib
def hash_dict(d):
    return hashlib.sha256(str(d).encode('utf-8')).hexdigest()

class EEGDataset(object):
    def get_name(self):
        raise NotImplementedError()
        #should return the name of the dataset

    def subjects(self):
        raise NotImplementedError()
        #should return a list of available subjects in dataset (string name)

    def get_subject_data(self):
        raise NotImplementedError()
        #should return mne raw array for this subject

class P300Dataset(object):
    def __init__(self, X, y, y_stim, y_names, y_stim_names, info, name='no_name'):
        assert len(X) == len(y) == len(y_stim)
        self.name = name
        self.info = info
        self.X = X
        self.y = y
        self.y_stim = y_stim
        self.y_names = y_names
        self.y_stim_names = y_stim_names

    def __repr__(self):
        return '{} containing :\nCounter: {}\nX: {}\ny: {}\ny_names: {}\ny_stim: {}\ny_stim_names: {}\n'.format(self.name, dict(Counter(self.y)), self.X.shape, self.y.shape, self.y_names, self.y_stim.shape, self.y_stim_names)

    def split(self, frac=0.8, split_names=(None, None)):
        split1_name = split_names[0] if split_names[0] else 'split1_from_{}'.format(self.name)
        split2_name = split_names[1] if split_names[1] else 'split2_from_{}'.format(self.name)
        length = self.X.shape[0]
        split_index = int(length * frac)
        split1 = P300Dataset(self.X[:split_index], self.y[:split_index], self.y_stim[:split_index], self.y_names, self.y_stim_names, self.info, split1_name)
        split2 = P300Dataset(self.X[split_index:], self.y[split_index:], self.y_stim[split_index:], self.y_names, self.y_stim_names, self.info, split2_name)
        return split1, split2

    def select(self, indices, subset_name=None):
        subset_name = subset_name if subset_name else 'subset_from_{}'.format(self.name)
        return P300Dataset(self.X[indices], self.y[indices], self.y_stim[indices], self.y_names, self.y_stim_names, self.info, subset_name)

    def split_by_indices(self, indices, split_names=(None, None)):
        split1_name = split_names[0] if split_names[0] else 'split1_from_{}'.format(self.name)
        split2_name = split_names[1] if split_names[1] else 'split2_from_{}'.format(self.name)
        indices = list(set(list(indices)))
        length = self.X.shape[0]
        indices_complement = [index for index in range(length) if index not in indices]
        return self.select(indices, subset_name=split1_name), self.select(indices_complement, subset_name=split2_name)

    def _duplicate_epoch(self, epoch_index, repeat=1):
        stacked_X = np.array([self.X[epoch_index],]*repeat)
        stacked_y = np.array([self.y[epoch_index],]*repeat)
        stacked_y_stim = np.array([self.y_stim[epoch_index],]*repeat)
        self.X = np.vstack((self.X, stacked_X))
        self.y = np.concatenate((self.y, stacked_y), axis=0)
        self.y_stim = np.concatenate((self.y_stim, stacked_y_stim), axis=0)
        return self

    def shuffle(self, random_state=None):
        self.X, self.y, self.y_stim = sklearn.utils.shuffle(self.X, self.y, self.y_stim, random_state=random_state)
        return self

    def clone(self):
        return P300Dataset(self.X, self.y, self.y_stim, self.y_names, self.y_stim_names, self.info, self.name)

    def sample(self, sampler, shuffle=True): #http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html#module-imblearn.under_sampling
        assert isinstance(sampler, imblearn.base.BaseSampler)
        first_dim, second_dim, third_dim = self.X.shape
        X_reshaped = self.X.reshape(first_dim, second_dim*third_dim)
        ly = len(self.y)
        index_hash = {hash_dict(item):index for index, item in enumerate(X_reshaped)}
        X_resampled, y_resampled, *selected_indices = sampler.fit_sample(X_reshaped, self.y)
        if X_resampled.shape[0] > X_reshaped.shape[0]:
            index_hash_resampled = {index:hash_dict(item) for index, item in enumerate(X_resampled)}

            print(X_reshaped.shape)
            print(X_resampled.shape)
            sampled = self.clone()
            for h, index in index_hash.items():
                c = list(index_hash_resampled.values()).count(h)
                if c > 1:
                    sampled._duplicate_epoch(index, repeat=c-1)
            if shuffle: sampled.shuffle()
            return sampled
        else:
            if not selected_indices: raise ValueError('sampler must return indices to apply sampler')
            split1, split2 = self.split_by_indices(selected_indices[0], split_names=('sampled_from_{}'.format(self.name), 'excluded_from_{}'.format(self.name)))
            if shuffle:
                split1.shuffle()
                split2.shuffle()
            return split1, split2

    def transform(self, transformer):
        assert isinstance(transformer, datasets.transformers.Transformer)
        ch_names = self.info.get('ch_names', [])
        ch_pos_map = dict(self.info.get('ch_pos', None))
        self.X, self.y = transformer.transform(self.X, self.y, ch_names=ch_names, ch_pos_map=ch_pos_map)

    @staticmethod
    def to_memmap(dataset, dir, filename):
        filename_data = '{}_data.mem'.format(filename)
        filename_target = '{}_target.mem'.format(filename)
        memmap_dict = {
            'data_shape': dataset.X.shape,
            'target_shape': dataset.y.shape
        }
        fp_data = np.memmap(os.path.join(dir, filename_data), dtype='float32', mode='w+', shape=dataset.X.shape)
        fp_data[:] = dataset.X
        fp_target = np.memmap(os.path.join(dir, filename_target), dtype=dataset.y.dtype, mode='w+', shape=dataset.y.shape)
        fp_target[:] = dataset.y
        memmap_filename = '{}_memmap_dict.json'.format(filename)
        f = open(os.path.join(dir, memmap_filename), "w")
        f.write(json.dumps(memmap_dict, indent=4, sort_keys=True))
        f.close()


    @staticmethod
    def resample(a, up=True):
        counter = dict(Counter(a.y))
        class_nb = max(list(counter.values())) if up else min(list(counter.values()))
        target_indices_majority = np.where(a.y == max(counter, key=counter.get))[0]

        majority, minority = a.split_by_indices(target_indices_majority, split_names=('target', 'non-target'))

        resampled = minority if up else majority
        to_concat = majority if up else minority
        resampled.X, resampled.y, resampled.y_stim = sklearn.utils.resample(resampled.X, resampled.y, resampled.y_stim, n_samples=class_nb, replace=up)

        concatenated = P300Dataset.concat(resampled, to_concat)
        concatenated.name = a.name
        return concatenated

    @staticmethod
    def concat(a, b):
        X = np.concatenate((a.X, b.X), axis=0)
        y = np.concatenate((a.y, b.y), axis=0)
        y_stim = np.concatenate((a.y_stim, b.y_stim), axis=0)
        concatenated = a.clone()
        concatenated.X = X
        concatenated.y = y
        concatenated.y_stim = y_stim
        return concatenated

    @staticmethod
    def concat_inplace(a, b):
        a.X = np.concatenate((a.X, b.X), axis=0)
        a.y = np.concatenate((a.y, b.y), axis=0)
        a.y_stim = np.concatenate((a.y_stim, b.y_stim), axis=0)
        b = None
        gc.collect()


    @staticmethod
    def concat_all(datasets):
        final_dataset = datasets[0]
        for dataset in datasets[1:]:
            P300Dataset.concat_inplace(final_dataset, dataset)
        return final_dataset

    @staticmethod
    def have_duplicates(a, b):
        assert isinstance(a, P300Dataset) and isinstance(b, P300Dataset)
        a_hash = [hash_dict(row) for row in a.X]
        b_hash = [hash_dict(row) for row in b.X]
        a_in_b = [h for h in a_hash if h in b_hash]
        b_in_a = [h for h in b_hash if h in a_hash]
        return bool(a_in_b) or bool(b_in_a)

    @staticmethod
    def from_mne_epoched(mne_epoched):
        assert isinstance(mne_epoched, mne.Epochs)
        X = (mne_epoched.get_data() * 1e6).astype(np.float32)
        y = (mne_epoched.events[:,2]).astype(np.int64)
        y_names = mne_epoched.info['y_names']
        y_stim = mne_epoched.info['y_stim']
        y_stim_names = mne_epoched.info['y_stim_names']
        info = {
            'nb_repetitions_by_item': mne_epoched.info['nb_repetitions_by_item'],
            # 'nb_repetitions': 10,
            'nb_items': mne_epoched.info['nb_items'],
            'repetition_length': mne_epoched.info['repetition_length'],
            'vocab_length': mne_epoched.info['vocab_length'],
            'sfreq': mne_epoched.info['sfreq'],
            'tmin': mne_epoched.info['tmin'],
            'tmax': mne_epoched.info['tmax'],
            'ch_names': mne_epoched.info['ch_names'],
            'chs': mne_epoched.info['chs'],
            'ch_pos': mne_epoched.info['ch_pos'],
            'exclude_channels': mne_epoched.info['exclude_channels'],
            'events': mne_epoched.events
        }
        return P300Dataset(X, y, y_stim, y_names, y_stim_names, info, mne_epoched.info['name'])
