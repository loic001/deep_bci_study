import json
import os
import argparse
import sys

from shutil import copyfile
from natsort import natsorted
from functools import reduce
from operator import add

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(8)

from torch.utils.data import Dataset, DataLoader, TensorDataset

import skorch
from skorch import NeuralNet
from skorch.dataset import CVSplit
from torch_ext.datasets import MemmapDataset

from deep_models_def import get_model_config_by_name_and_scenario

from utils.db import ObjectSaverJoblib as ObjectSaver
from utils.testers import P300DatasetTesterSpelling

from utils.callbacks import MyCheckpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', required=True)
    # parser.add_argument('--output-dir', help='output dir',
    #                     type=str, required=True)
    parser.add_argument("--model-name", help='name of the model',
                        type=str, default='cnn2d_tdb_topo')
    parser.add_argument("--output-dir-base", help='output dir',
                        type=str, default='outputs')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--max-epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 250)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L', help='lr')

    args = parser.parse_args()

    with open(args.scenario, encoding='utf-8') as f:
        scenario = json.loads(f.read())

    lr = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    model_name = args.model_name

    model = get_model_config_by_name_and_scenario(model_name, scenario)
    output_dir_base = args.output_dir_base
    name = scenario['name']
    output_dir = os.path.join(output_dir_base, name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copyfile(args.scenario, os.path.join(output_dir, 'scenario.json'))

    train_keys = scenario['train']
    memmap_datasets = [MemmapDataset(dir=d.get('memmap_dir'), name=d.get('memmap_name')) for d in train_keys]
    train_dataset = reduce(add, memmap_datasets)

    cv = CVSplit(0.2, random_state=0)
    train_dataset, valid_dataset = cv(train_dataset)

    def train_split_func(train_dataset, _):
        return train_dataset, valid_dataset

    checkpoints = []
    def last_checkpoint(checkpoint):
        checkpoints.append(checkpoint)

    skorch_config = {
        'module': model['class_name'],
        'optimizer': torch.optim.RMSprop,
        'optimizer__momentum': 0.9,
        'optimizer__weight_decay': 0.0005,
        'callbacks': [
            # ('auc', skorch.callbacks.EpochScoring(scoring='roc_auc', lower_is_better=False, use_caching=False)),
            ('checkpoints', MyCheckpoint(os.path.join(output_dir, model_name+'__{last_epoch[epoch]}.pt'), callback_func=last_checkpoint)),
            # ('layer_data_logger_1', LayerDataViz(layer_name='sub_module.flatten', callback_func=partial(viz.layer_data_viz, {'name': 'toto'}))),
            # ('epoch_summary', EpochSummary(callback_func=partial(viz.epoch_summary, None))),
            # ('params_epoch_end', ParamsViz(callback_func=partial(viz.params_viz, None)))
        ],
        # 'optimizer__momentum': 0.9,
        'criterion': torch.nn.NLLLoss,
        # 'criterion__weight': torch.FloatTensor([1, 29]),
        'iterator_train__num_workers': 16,
        'iterator_train__shuffle': True,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'train_split': train_split_func,
        'warm_start': True
    }
    skorch_config = {**skorch_config, **model['params']}
    skorch_config['lr'] = lr
    print(skorch_config)

    net = NeuralNet(**skorch_config)

    from utils.viz import summary
    net.fit(train_dataset, y=None)
    print(summary(net.module_))
    net.load_params(checkpoints[-1])
    # net.save_params(os.path.join(output_dir, model_name+'_params'))
    # net.save_history(os.path.join(output_dir, model_name+'_history.json'))
    test_info = scenario['test'][0]
    p300_dataset_dir = test_info['p300_dataset_dir']
    p300_dataset_dir = p300_dataset_dir.replace('/mnt/data/loic/data', '/data')
    p300_dataset_key = test_info['p300_dataset_key']
    p300_dataset_name = test_info['p300_dataset_name']
    with open(os.path.join(p300_dataset_dir, 'data_dict.json'), encoding='utf-8') as f:
        data_dict = json.loads(f.read())
    saver = ObjectSaver(p300_dataset_dir)
    p300_dataset = saver.load(data_dict[p300_dataset_name][p300_dataset_key])
    print(p300_dataset)
    print('------')
    print(p300_dataset.info['nb_items'])
    print('------')
    tester = P300DatasetTesterSpelling(net, p300_dataset)
    results = tester.run().render()
    print(pd.DataFrame(results))
