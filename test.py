import json
import os
import argparse
import sys
import re

import logging
logging.basicConfig(level=logging.DEBUG)

import multiprocessing

import pickle
import numpy as np
import pandas as pd
import torch
import glob
from functools import reduce
from natsort import natsorted

from deep_models_def import get_model_config_by_name_and_scenario as get_model_config_deep
from riemann_models_def import get_model_config_by_name_and_scenario as get_model_config_riemann
from deep_models_def import deep_models_2d
from deep_models_def import deep_models_3d
deep_models = deep_models_2d + deep_models_3d
from riemann_models_def import riemann_models

from utils.db import ObjectSaverJoblib as ObjectSaver
from utils.testers import P300DatasetTesterSpelling

import skorch
from skorch import NeuralNet

output_dir = '/home/lolo/projects/deep_bci_study/outputs'
PARAMS_FILE = 'params'
HISTORY_FILE = 'history.json'
SCENARIO_FILE = 'scenario.json'
exclude_dirs = ['log']

def clean_expe_out(expe_out, root_dir):
    d = {}
    for expe, expe_struct in expe_out.items():
        if expe in exclude_dirs:
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
print(expe_out)
expe_out = clean_expe_out(expe_out, output_dir)

print(json.dumps(expe_out, indent=2))

def spelling_test(params):
    print('spelling test')
    expe, subject, model_name, net, p300_dataset, log_proba = params
    tester = P300DatasetTesterSpelling(net, p300_dataset, log_proba=log_proba)
    results = tester.run().render()
    recs = []
    for res in results:
        rec = {
            'expe': expe,
            'subject': subject,
            'model': model_name,
            'rep': res['after_x_rep'],
            'acc': res['acc']
        }
        recs.append(rec)
    return recs

print('testing...')

records = []
errors = []
for expe, subjects in expe_out.items():
    print(expe)
    pool_map_params = []
    for subject, subjects_data in subjects.items():
        print(subject)
        scenario_file = subjects_data['scenario']
        print(scenario_file)
        with open(scenario_file, encoding='utf-8') as f:
            scenario = json.loads(f.read())
        test_info = scenario['test'][0]
        p300_dataset_dir = test_info['p300_dataset_dir']
        p300_dataset_dir = p300_dataset_dir.replace('/mnt/data/loic/data', '/data')
        p300_dataset_key = test_info['p300_dataset_key']
        p300_dataset_name = test_info['p300_dataset_name']
        with open(os.path.join(p300_dataset_dir, 'data_dict.json'), encoding='utf-8') as f:
            data_dict = json.loads(f.read())
        saver = ObjectSaver(p300_dataset_dir)
        p300_dataset = saver.load(data_dict[p300_dataset_name][p300_dataset_key])
        # print(p300_dataset)
        #deep models
        for model_name, data in subjects_data['models'].items():
            checkpoint = data.get('checkpoints', [None])[-1]
            history_file = data.get('history')
            if not history_file or not checkpoint:
                errors.append({
                    'expe': expe,
                    'subject': subject,
                    'model': model_name,
                })
                continue
            print(checkpoint)
            # print(history_file)
            model = get_model_config_deep(model_name, scenario)
            # print(model)
            skorch_config = {
                'module': model['class_name'],
                'criterion': torch.nn.NLLLoss
            }
            skorch_config = {**skorch_config, **model['params']}
            net = NeuralNet(**skorch_config)
            net.initialize()
            net.load_params(checkpoint)
            net.load_history(history_file)
            pool_map_params.append((expe, subject, model_name, net, p300_dataset, True))

        #riemann models
        for model_name, data in subjects_data['riemann_models'].items():
            checkpoint = data.get('checkpoint')
            net = pickle.load(open(checkpoint, "rb"))
            pool_map_params.append((expe, subject, model_name, net, p300_dataset, False))
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()/2))
    results = pool.map(spelling_test, pool_map_params)
    for res in results:
        for _res in res:
            records.append(_res)

df = pd.DataFrame(records)
df.to_pickle(os.path.join(output_dir, 'acc.pkl'))

df_err = pd.DataFrame(errors)
df_err.to_pickle(os.path.join(output_dir, 'err.pkl'))
