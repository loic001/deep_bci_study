import json
import os
import argparse
import sys
import re

import logging
logging.basicConfig(level=logging.DEBUG)

import pickle
import numpy as np
import pandas as pd
import torch
import glob
import argparse
from functools import reduce
from natsort import natsorted

from deep_models_def import get_model_config_by_name_and_scenario as get_model_config_deep
from riemann_models_def import get_model_config_by_name_and_scenario as get_model_config_riemann

from deep_models_def import deep_models_2d
from deep_models_def import deep_models_3d
deep_models_def = deep_models_2d + deep_models_3d

from riemann_models_def import riemann_models

from utils.db import ObjectSaverJoblib as ObjectSaver
from utils.testers import P300DatasetTesterSpelling

import skorch
from skorch import NeuralNet


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", help='output dir', type=str, required=True)
args = parser.parse_args()

# afterany

output_dir = args.output_dir
print('output_dir: {}'.format(output_dir))
PARAMS_FILE = 'params'
HISTORY_FILE = 'history.json'
SCENARIO_FILE = 'scenario.json'
exclude_dirs_regex = ['log', '(.*).pkl']


def clean_expe_out(expe_out, root_dir):
    d = {}
    for expe, expe_struct in expe_out.items():
        if(re.match("(" + ")|(".join(exclude_dirs_regex) + ")", expe)):
            continue
        d_sub = {}
        for subject, file_dict in expe_struct.items():
            checkpoints_sorted = natsorted([os.path.join(root_dir, expe, subject, f) for f in list(file_dict.keys())])
            deep_models = []
            for ch in checkpoints_sorted:
                res = re.findall('{}/(.*)__[0-9]*\.pt'.format(subject), ch)
                if res:
                    deep_models.append(res[0])
            deep_models = list(set(deep_models))
            riemann_models = []
            for ch in checkpoints_sorted:
                print(ch)
                res = re.findall('{}/model__(.*).pkl'.format(subject), ch)
                print(res)
                if res:
                    riemann_models.append(res[0])
            riemann_models = list(set(riemann_models))
            # riemann_models = list(set([re.findall('{}/model__(.*).pkl'.format(subject), ch)[0] for ch in list(file_dict.keys())]))
            d_sub[subject] = {
                # 'checkpoints': natsorted([os.path.join(root_dir, expe, subject, f) for f in list(file_dict.keys()) if f[-3:] == '.pt']),
                # 'params': PARAMS_FILE,
                # 'history': HISTORY_FILE,
                'models': {},
                'riemann_models': {},
                'scenario': os.path.join(root_dir, expe, subject, 'scenario.json')
            }
            for model in riemann_models:
                d_model = {
                    'checkpoint': os.path.join(root_dir, expe, subject, 'model__{}.pkl'.format(model))
                }
                d_sub[subject]['riemann_models'][model]=d_model
            for model in deep_models:
                d_model = {
                    'checkpoints': natsorted([os.path.join(root_dir, expe, subject, f) for f in list(file_dict.keys()) if re.match('{}__[0-9]*.pt'.format(model), f)])
                }
                if [os.path.join(root_dir, expe, subject, f) for f in list(file_dict.keys()) if re.match('{}_{}'.format(model, HISTORY_FILE), f)]:
                    d_model['history']=os.path.join(root_dir, expe, subject, '{}_{}'.format(model, HISTORY_FILE))
                if [os.path.join(root_dir, expe, subject, f) for f in list(file_dict.keys()) if re.match('{}_{}'.format(model, PARAMS_FILE), f)]:
                    d_model['params']=os.path.join(root_dir, expe, subject, '{}_{}'.format(model, PARAMS_FILE))
                d_sub[subject]['models'][model]=d_model
        d[expe] = d_sub
    return d



def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir
    """
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir

dir_struct = get_directory_structure(output_dir)
expe_out = dir_struct[list(dir_struct.keys())[0]]
expe_out = clean_expe_out(expe_out, output_dir)
#
print(json.dumps(expe_out, indent=2))

records = []
errors = []
for expe, subjects in expe_out.items():
    print(expe)
    pool_map_params = []
    for subject, subjects_data in subjects.items():
        print(subject)
        for model_name, data in subjects_data['models'].items():
            history_file = data.get('history')
            if not history_file:
                errors.append({
                    'expe': expe,
                    'subject': subject,
                    'model': model_name,
                })
                continue
            data = json.load(open(history_file, 'r'))
            best = [item for item in data if item['valid_loss_best'] == True][-1]
            print('epoch: {}'.format(best['epoch']))
            print('train_loss: {}'.format(best['train_loss']))
            print('valid_loss: {}'.format(best['valid_loss']))
            print('total_epoch: {}'.format(data[-1]['epoch']))
            records.append({
                'subject': subject,
                'epoch': best['epoch'],
                'train_loss': best['train_loss'],
                'valid_loss': best['valid_loss'],
                'total_epoch': data[-1]['epoch'],
                'expe': expe,
                'subject': subject,
                'model': model_name,
                })

df = pd.DataFrame(records)
df.to_pickle(os.path.join(output_dir, 'valid_loss.pkl'))


df_err = pd.DataFrame(errors)
df_err.to_pickle(os.path.join(output_dir, 'valid_loss_err.pkl'))
