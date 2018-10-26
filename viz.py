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

from utils.gradients import SmoothGrad, VanillaGrad

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", help='output dir', type=str, required=True)
args = parser.parse_args()

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
print(expe_out)
expe_out = clean_expe_out(expe_out, output_dir)

print(json.dumps(expe_out, indent=2))

import multiprocessing

def spelling_test(params):
    print('spelling test')
    expe, subject, model_name, net, p300_dataset, log_proba = params
    model = net.module_
    vanilla_grad = VanillaGrad(pretrained_model=model, cuda=False)
    all = []
    data = p300_dataset.X
    target = p300_dataset.select(p300_dataset.y == 1).shuffle(random_state=2).X
    non_target = p300_dataset.select(p300_dataset.y == 0).shuffle(random_state=2).X
    non_target = non_target[:len(target)]
    for id, item in enumerate(non_target):
        sample = np.expand_dims(item, axis=0)
        smooth_saliency = vanilla_grad(sample, index=0)
        all.append(np.expand_dims(smooth_saliency, axis=0))
    smooth_grad_non_target = np.concatenate(all)
    vanilla_grad = VanillaGrad(pretrained_model=model, cuda=False)
    all = []
    data = p300_dataset.X
    for id, item in enumerate(target):
        sample = np.expand_dims(item, axis=0)
        smooth_saliency = vanilla_grad(sample, index=1)
        all.append(np.expand_dims(smooth_saliency, axis=0))
    smooth_grad_target = np.concatenate(all)
    print('finish')
    # smooth_grad = SmoothGrad(pretrained_model=model, cuda=False, magnitude=True, n_samples=15)
    # all = []
    # data = p300_dataset.X
    # for item in data:
    #     sample = np.expand_dims(item, axis=0)
    #     smooth_saliency = smooth_grad(sample, index=1)
    #     all.append(np.expand_dims(smooth_saliency, axis=0))
    # smooth_grad_target = np.concatenate(all)
    #
    rec = {
        'expe': expe,
        'subject': subject,
        'model': model_name,
        'smooth_grad_target': smooth_grad_target,
        'smooth_grad_non_target': smooth_grad_non_target

    }
    return rec



print('testing......')

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
            print(model_name)
            # if model_name not in ['cnn1d_t2']:
            #     continue
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
    pool = multiprocessing.Pool(processes=10)
    results = pool.map(spelling_test, pool_map_params)
    for res in results:
        records.append(res)

pickle.dump(records, open(os.path.join(output_dir, 'viz.pkl'), 'wb'))
