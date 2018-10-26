import json
import os
import argparse
import sys

import logging
logging.basicConfig(level=logging.DEBUG)

from shutil import copyfile
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset, DataLoader, TensorDataset

from skorch.dataset import CVSplit
from torch_ext.datasets import MemmapDataset

from riemann_models_def import get_model_config_by_name_and_scenario

from utils.db import ObjectSaverJoblib as ObjectSaver
from utils.testers import P300DatasetTesterSpelling

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True)
    parser.add_argument("--model-name", help='name of the model', type=str, default='riemann_xdawn_tang_log')
    parser.add_argument("--output-dir-base", help='output dir', type=str, default='outputs')
    args = parser.parse_args()

    with open(args.scenario, encoding='utf-8') as f:
        scenario = json.loads(f.read())

    output_dir_base = args.output_dir_base
    name = scenario['name']
    output_dir = os.path.join(output_dir_base, name)
    model_name = args.model_name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copyfile(args.scenario, os.path.join(output_dir, 'scenario.json'))
    memmap_datasets = [MemmapDataset(dir=d.get('memmap_dir'), name=d.get('memmap_name')) for d in scenario['train']]
    train_dataset = reduce(add, memmap_datasets)
    train_dataset, valid_dataset = CVSplit(0.2, random_state=0)(train_dataset)
    X_train = np.concatenate([np.expand_dims(item[0], 0) for item in train_dataset])
    y_train = np.array([item[1] for item in train_dataset])
    X_valid = np.concatenate([np.expand_dims(item[0], 0) for item in valid_dataset])
    y_valid = np.array([item[1] for item in valid_dataset])
    X = np.concatenate((X_train, X_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)
    model_config = get_model_config_by_name_and_scenario(model_name, scenario)
    model = model_config['class_name'](model_config['params'])
    model.fit(X, y)
    pickle.dump(model, open(os.path.join(output_dir, 'model__{}.pkl'.format(model_name)), "wb"))

    #test
    test_info = scenario['test'][0]
    p300_dataset_dir = test_info['p300_dataset_dir']
    p300_dataset_dir = p300_dataset_dir.replace('/mnt/data/loic/data', '/data')
    p300_dataset_key = test_info['p300_dataset_key']
    p300_dataset_name = test_info['p300_dataset_name']
    with open(os.path.join(p300_dataset_dir, 'data_dict.json'), encoding='utf-8') as f:
        data_dict = json.loads(f.read())
    saver = ObjectSaver(p300_dataset_dir)
    p300_dataset = saver.load(data_dict[p300_dataset_name][p300_dataset_key])
    tester = P300DatasetTesterSpelling(model, p300_dataset, log_proba=False)
    results = tester.run().render()
    print(pd.DataFrame(results))
