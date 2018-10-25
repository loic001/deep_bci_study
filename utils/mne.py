import logging
import mne
import numpy as np
from copy import deepcopy

def raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels):
        logging.info('Building epochs : name: {}, tmin: {}, tmax: {}, exclude_channels: {}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels)))
        return mne.Epochs(raw_mne, raw_mne.info['y'], dict(target=1, non_target=0), tmin=tmin, tmax=tmax, preload=True).pick_types(eeg=True, exclude=exclude_channels)

def cache_or_create_raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels, cache=False, cache_dir='/tmp'):
    epoched = None
    if cache:
        assert cache_dir is not None
        db_name = os.path.join(cache_dir, 'datasets_cache')
        db_key = '{}_{}_{}_{}_{}_{}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels), str(raw_mne.info['lowpass']), str(raw_mne.info['highpass']))
        db_key = hash_string(db_key)
        with shelve.open(db_name) as db:
            try:
                epoched=db[db_key]
                logging.info('Epochs successfully recovered : name: {}, tmin: {}, tmax: {}, exclude_channels: {}'.format(raw_mne.info['name'], str(tmin), str(tmax), str(exclude_channels)))
            except KeyError:
                logging.info('Requested epochs do not exist in cache')
                epoched = raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels)
                db[db_key]=epoched
    else:
        epoched = raw_to_epoched_mne(raw_mne, tmin, tmax, exclude_channels)
    return epoched

def concat_raw_arrays(a, b):
    assert a.info['p300_speller_type'] == b.info['p300_speller_type']
    assert a.info['repetition_length'] == b.info['repetition_length']
    assert a.info['vocab_length'] == b.info['vocab_length']
    info = deepcopy(a.info)
    a_x_length = a.get_data().shape[1]

    #inplace modify b events
    b.info['y'][:,0] = b.info['y'][:,0] + a_x_length
    info['y'] = np.concatenate((a.info['y'], b.info['y']))

    info['nb_repetitions_by_item'] = np.concatenate((a.info['nb_repetitions_by_item'], b.info['nb_repetitions_by_item']))
    info['y_stim'] = np.concatenate((a.info['y_stim'], b.info['y_stim']))
    info['name'] = '{}_{}'.format(a.info['name'], b.info['name'])
    info['nb_items'] = a.info['nb_items'] + b.info['nb_items']

    data = np.concatenate((a.get_data(), b.get_data()), axis=1)
    return mne.io.RawArray(data, info, verbose='WARNING')

def concat_raw_arrays_list(raw_arrays):
    assert len(raw_arrays) > 0
    raw_array = raw_arrays[0]
    for _raw_array in raw_arrays[1:]:
        raw_array = concat_raw_arrays(raw_array, _raw_array)
    return raw_array

class P300Sequencer(object):
    def __init__(self, mne_raw_array, fill_value='auto'):
        self.mne_raw_array = mne_raw_array
        self.y_index = self.mne_raw_array.info['y'][:,0]
        self.y_target = self.mne_raw_array.info['y'][:,2]
        self.y_stim = self.mne_raw_array.info['y_stim']
        self.repetition_length = self.mne_raw_array.info['repetition_length']
        self.data, self.times = self.mne_raw_array[:]

        self.fill_value = fill_value

        self.max_index = self.y_index.shape[0] // self.repetition_length

        self.reset()

    def next(self):
        index_cursor_mod = self._next_index_cursor % self.max_index
        self._next_index_cursor += 1
        return self.get(index_cursor_mod)

    def reset(self):
        self._next_index_cursor = 0

    def get(self, index):
        assert self.max_index
        start_sequence = index*self.repetition_length
        end_sequence= start_sequence+self.repetition_length
        sequence_indexes = self.y_index[start_sequence:end_sequence]

        soa_timestep = list(np.diff(sequence_indexes))
        mean_soa = np.asscalar(np.mean(soa_timestep))
        mean_soa_int = int(mean_soa)

        soa_timestep.append(mean_soa_int)

        start_sequence_index, end_sequence_index = sequence_indexes[0], sequence_indexes[-1]
        end_sequence_index = end_sequence_index + mean_soa_int

        data, times = self.mne_raw_array[:, start_sequence_index:end_sequence_index]

        items_indexed = self.y_stim[start_sequence:end_sequence]
        target_indexed = self.y_target[start_sequence:end_sequence]
        target_item = np.asscalar(items_indexed[target_indexed == 1])

        items_sequenced = []
        for index, item in enumerate(list(items_indexed)):
            items_sequenced.append(item)
            if index < len(soa_timestep):
                for _ in range(soa_timestep[index]-1):
                    if self.fill_value != 'auto':
                        items_sequenced.append(self.fill_value)
                    else:
                        items_sequenced.append(item)

        items_sequenced = np.array(items_sequenced)
        return data, target_item, items_indexed, items_sequenced
