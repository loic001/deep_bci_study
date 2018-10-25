import json
import os
import sys
import argparse
from itertools import permutations

parser = argparse.ArgumentParser()
parser.add_argument('--base-path',  help='data base path', type=str, required=True)
parser.add_argument("--output-dir", help='output dir', type=str, default='/tmp/scenarios')
args = parser.parse_args()
output_dir = args.output_dir
DATA_BASE_PATH = args.base_path

meeg_prefix = 'P300_SPELLER_MEEG'
rsvp_prefix = 'RSVP_COLOR_116'

rsvp_subjects = ['VPgcc', 'VPfat', 'VPgcb', 'VPgcg', 'VPgcd', 'VPgcf', 'VPgch',
                'VPiay', 'VPicn', 'VPicr', 'VPpia']
meeg_subjects = ['S02', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S19',
                'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28']

cum_sets = ['meeg48', 'rsvp48']

datasets = {
    'rsvp': {
        'memmap_topo_dir': os.path.join(DATA_BASE_PATH, 'rsvp_topo/memmap'),
        'memmap_dir': os.path.join(DATA_BASE_PATH, 'rsvp/memmap'),
        'p300_dataset_dir': os.path.join(DATA_BASE_PATH, 'rsvp'),
        'p300_dataset_dir_topo': os.path.join(DATA_BASE_PATH, 'rsvp_topo'),
        'prefix': rsvp_prefix,
        'subjects': rsvp_subjects,
        'params': {
            'n_chans': 55
        }
    },
    'meeg': {
        'memmap_topo_dir': os.path.join(DATA_BASE_PATH, 'meeg_topo/memmap'),
        'memmap_dir': os.path.join(DATA_BASE_PATH, 'meeg/memmap'),
        'p300_dataset_dir': os.path.join(DATA_BASE_PATH, 'meeg'),
        'p300_dataset_dir_topo': os.path.join(DATA_BASE_PATH, 'meeg_topo'),
        'prefix': meeg_prefix,
        'subjects': meeg_subjects,
        'params': {
            'n_chans': 56
        }
    },
    'meeg48': {
        'memmap_topo_dir': os.path.join(DATA_BASE_PATH, 'meeg48_topo/memmap'),
        'memmap_dir': os.path.join(DATA_BASE_PATH, 'meeg48/memmap'),
        'p300_dataset_dir': os.path.join(DATA_BASE_PATH, 'meeg48'),
        'p300_dataset_dir_topo': os.path.join(DATA_BASE_PATH, 'meeg48_topo'),
        'prefix': meeg_prefix,
        'subjects': meeg_subjects,
        'params': {
            'n_chans': 48
        }
    },
    'rsvp48': {
        'memmap_topo_dir': os.path.join(DATA_BASE_PATH, 'rsvp48_topo/memmap'),
        'memmap_dir': os.path.join(DATA_BASE_PATH, 'rsvp48/memmap'),
        'p300_dataset_dir': os.path.join(DATA_BASE_PATH, 'rsvp48'),
        'p300_dataset_dir_topo': os.path.join(DATA_BASE_PATH, 'rsvp48_topo'),
        'prefix': rsvp_prefix,
        'subjects': rsvp_subjects,
        'params': {
            'n_chans': 48
        }
    }
}

def build_within_scenario(prefix, subjects, dir, p300_dataset_dir, memmap_dir, dataset_name, params):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for subject in subjects:
        filename = '{}.json'.format(subject)
        content = {
            'params': params,
            'name': dataset_name + '/' + subject,
            'train': [
                {
                    'memmap_dir': memmap_dir,
                    'memmap_name': '{}_{}_calib'.format(prefix, subject),
                    'p300_dataset_dir': p300_dataset_dir,
                    'p300_dataset_name': '{}_{}'.format(prefix, subject),
                    'p300_dataset_key': 'calib_dataset'
                }
            ],
            'test': [
                {
                    'memmap_dir': memmap_dir,
                    'memmap_name': '{}_{}'.format(prefix, subject),
                    'p300_dataset_dir': p300_dataset_dir,
                    'p300_dataset_name': '{}_{}'.format(prefix, subject),
                    'p300_dataset_key': 'dataset'
                }
            ]
        }
        f = open(os.path.join(dir, filename),"w")
        f.write(json.dumps(content, indent=4))
        f.close()

def build_transfer_scenario(prefix, subjects, dir, p300_dataset_dir, memmap_dir, dataset_name, params):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for subject in subjects:
        filename = '{}.json'.format(subject)
        all_but_one_calib = [{'memmap_dir': memmap_dir, 'memmap_name': '{}_{}_calib'.format(prefix, _subject), 'p300_dataset_dir': p300_dataset_dir, 'p300_dataset_name': '{}_{}'.format(prefix, _subject), 'p300_dataset_key': 'calib_dataset'} for _subject in subjects if _subject != subject]
        all_but_one = [{'memmap_dir': memmap_dir, 'memmap_name': '{}_{}'.format(prefix, _subject), 'p300_dataset_dir': p300_dataset_dir,'p300_dataset_name': '{}_{}'.format(prefix, _subject), 'p300_dataset_key': 'dataset'} for _subject in subjects if _subject != subject]
        content = {
            'params': params,
            'name': dataset_name + '_transfer/' + subject,
            'train': all_but_one_calib + all_but_one,
            'test': [
                {
                    'memmap_dir': memmap_dir,
                    'memmap_name': '{}_{}'.format(prefix, subject),
                    'p300_dataset_dir': p300_dataset_dir,
                    'p300_dataset_name': '{}_{}'.format(prefix, subject),
                    'p300_dataset_key': 'dataset'
                }
            ]
        }
        f = open(os.path.join(dir, filename),"w")
        f.write(json.dumps(content, indent=4))
        f.close()



def build_cum_scenario(dir, prefix_1, subjects_1, p300_dataset_dir_1, memmap_dir_1, d1_k_1, params_1,
                       prefix_2, subjects_2, p300_dataset_dir_2, memmap_dir_2, d1_k_2, params_2):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for subject_1 in subjects_1:
        filename = '{}.json'.format(subject_1)
        all_but_one_calib_1 = [{'memmap_dir': memmap_dir_1, 'memmap_name': '{}_{}_calib'.format(prefix_1, _subject), 'p300_dataset_dir': p300_dataset_dir_1, 'p300_dataset_name': '{}_{}'.format(prefix_1, _subject), 'p300_dataset_key': 'calib_dataset'} for _subject in subjects_1 if _subject != subject_1]
        all_but_one_1 = [{'memmap_dir': memmap_dir_1, 'memmap_name': '{}_{}'.format(prefix_1, _subject), 'p300_dataset_dir': p300_dataset_dir_1,'p300_dataset_name': '{}_{}'.format(prefix_1, _subject), 'p300_dataset_key': 'dataset'} for _subject in subjects_1 if _subject != subject_1]
        all_but_one_calib_2 = [{'memmap_dir': memmap_dir_2, 'memmap_name': '{}_{}_calib'.format(prefix_2, _subject), 'p300_dataset_dir': p300_dataset_dir_2, 'p300_dataset_name': '{}_{}'.format(prefix_2, _subject), 'p300_dataset_key': 'calib_dataset'} for _subject in subjects_2]

        content = {
            'params': params_1,
            'name': '{}_cum_{}/{}'.format(d1_k_1, d1_k_2, subject_1),
            'train': all_but_one_calib_1 + all_but_one_1 + all_but_one_calib_2,
            'test': [
                {
                    'memmap_dir': memmap_dir_1,
                    'memmap_name': '{}_{}'.format(prefix_1, subject_1),
                    'p300_dataset_dir': p300_dataset_dir_1,
                    'p300_dataset_name': '{}_{}'.format(prefix_1, subject_1),
                    'p300_dataset_key': 'dataset'
                }
            ]
        }
        f = open(os.path.join(dir, filename),"w")
        f.write(json.dumps(content, indent=4))
        f.close()

def build_expe_transfer_scenario(dir, prefix_1, subjects_1, p300_dataset_dir_1, memmap_dir_1, d1_k_1, params_1,
                       prefix_2, subjects_2, p300_dataset_dir_2, memmap_dir_2, d1_k_2, params_2):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for subject_1 in subjects_1:
        filename = '{}.json'.format(subject_1)
        all_but_one_2 = [{'memmap_dir': memmap_dir_2, 'memmap_name': '{}_{}'.format(prefix_2, _subject), 'p300_dataset_dir': p300_dataset_dir_2,'p300_dataset_name': '{}_{}'.format(prefix_2, _subject), 'p300_dataset_key': 'dataset'} for _subject in subjects_2]
        all_but_one_calib_2 = [{'memmap_dir': memmap_dir_2, 'memmap_name': '{}_{}_calib'.format(prefix_2, _subject), 'p300_dataset_dir': p300_dataset_dir_2, 'p300_dataset_name': '{}_{}'.format(prefix_2, _subject), 'p300_dataset_key': 'calib_dataset'} for _subject in subjects_2]

        content = {
            'params': params_1,
            'name': '{}_expe_transfer_{}/{}'.format(d1_k_1, d1_k_2, subject_1),
            'train': all_but_one_2 + all_but_one_calib_2,
            'test': [
                {
                    'memmap_dir': memmap_dir_1,
                    'memmap_name': '{}_{}'.format(prefix_1, subject_1),
                    'p300_dataset_dir': p300_dataset_dir_1,
                    'p300_dataset_name': '{}_{}'.format(prefix_1, subject_1),
                    'p300_dataset_key': 'dataset'
                }
            ]
        }
        f = open(os.path.join(dir, filename),"w")
        f.write(json.dumps(content, indent=4))
        f.close()


for dataset_name, v in datasets.items():
    _prefix = v['prefix']
    _subjects = v['subjects']
    _memmap_dir = v['memmap_dir']
    _memmap_topo_dir = v['memmap_topo_dir']
    _p300_dataset_dir = v['p300_dataset_dir']
    _p300_dataset_dir_topo = v['p300_dataset_dir_topo']
    _params = v['params']
    _prefix = v['prefix']

    build_within_scenario(_prefix, _subjects, os.path.join(output_dir, dataset_name), _p300_dataset_dir, _memmap_dir, dataset_name, _params)
    build_within_scenario(_prefix, _subjects, os.path.join(output_dir, '{}_topo'.format(dataset_name)), _p300_dataset_dir_topo, _memmap_topo_dir, dataset_name+'_topo', _params)
    build_transfer_scenario(_prefix, _subjects, os.path.join(output_dir, '{}_transfer'.format(dataset_name)),_p300_dataset_dir,  _memmap_dir, dataset_name, _params)
    build_transfer_scenario(_prefix, _subjects, os.path.join(output_dir, '{}_topo_transfer'.format(dataset_name)), _p300_dataset_dir_topo, _memmap_topo_dir, dataset_name+'_topo', _params)

cum_datasets = {k:v for k,v in datasets.items() if k in cum_sets}
for (d1_k, d1_v), (d2_k, d2_v) in permutations(cum_datasets.items(), 2):
    _subjects_1 = d1_v['subjects']
    _memmap_dir_1 = d1_v['memmap_dir']
    _memmap_topo_dir_1 = d1_v['memmap_topo_dir']
    _p300_dataset_dir_1 = d1_v['p300_dataset_dir']
    _p300_dataset_dir_topo_1 = d1_v['p300_dataset_dir_topo']
    _params_1 = d1_v['params']
    _prefix_1 = d1_v['prefix']

    _subjects_2 = d2_v['subjects']
    _memmap_dir_2 = d2_v['memmap_dir']
    _memmap_topo_dir_2 = d2_v['memmap_topo_dir']
    _p300_dataset_dir_2 = d2_v['p300_dataset_dir']
    _p300_dataset_dir_topo_2 = d2_v['p300_dataset_dir_topo']
    _params_2 = d2_v['params']
    _prefix_2 = d2_v['prefix']
    build_cum_scenario(os.path.join(output_dir, '{}_cum_{}'.format(d1_k, d2_k)), _prefix_1, _subjects_1, _p300_dataset_dir_1,  _memmap_dir_1, d1_k, _params_1,
                        _prefix_2, _subjects_2,_p300_dataset_dir_2,  _memmap_dir_2, d2_k, _params_2)
    build_cum_scenario(os.path.join(output_dir, '{}_topo_cum_{}'.format(d1_k, d2_k)), _prefix_1, _subjects_1, _p300_dataset_dir_topo_1,  _memmap_topo_dir_1, d1_k+'_topo', _params_1,
                        _prefix_2, _subjects_2,_p300_dataset_dir_topo_2,  _memmap_topo_dir_2, d2_k, _params_2)
    build_expe_transfer_scenario(os.path.join(output_dir, '{}_expe_transfer_{}'.format(d1_k, d2_k)), _prefix_1, _subjects_1, _p300_dataset_dir_1,  _memmap_dir_1, d1_k, _params_1,
                        _prefix_2, _subjects_2,_p300_dataset_dir_2,  _memmap_dir_2, d2_k, _params_2)
    build_expe_transfer_scenario(os.path.join(output_dir, '{}_topo_expe_transfer_{}'.format(d1_k, d2_k)), _prefix_1, _subjects_1, _p300_dataset_dir_topo_1,  _memmap_topo_dir_1, d1_k+'_topo', _params_1,
                        _prefix_2, _subjects_2,_p300_dataset_dir_topo_2,  _memmap_topo_dir_2, d2_k, _params_2)
