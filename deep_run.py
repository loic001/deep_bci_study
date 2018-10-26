import json
import os
import argparse
import sys

from shutil import copyfile, move
from natsort import natsorted
from functools import reduce
from operator import add
import glob
import logging
import re
logging.basicConfig(level=logging.DEBUG)

import numpy as np
import torch
torch.set_num_threads(8)

import skorch
from skorch import NeuralNet
from skorch.dataset import CVSplit
from torch_ext.datasets import MemmapDataset

from torch.utils.data import Dataset, DataLoader, TensorDataset

from deep_models_def import get_model_config_by_name_and_scenario

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
    parser.add_argument('--continue-training', action='store_true', default=False,
                        help='init model with params and history file and continue training')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L', help='lr')

    args = parser.parse_args()

    with open(args.scenario, encoding='utf-8') as f:
        scenario = json.loads(f.read())

    lr = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    model_name = args.model_name
    continue_training = args.continue_training

    model = get_model_config_by_name_and_scenario(model_name, scenario)
    output_dir_base = args.output_dir_base
    name = scenario['name']
    output_dir = os.path.join(output_dir_base, name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    copyfile(args.scenario, os.path.join(output_dir, 'scenario.json'))

    old_history_file = None
    if continue_training:
        #looking for checkpoint to continue training
        checkpoints = natsorted(glob.glob(os.path.join(output_dir, '{}__[0-9]*.pt'.format(model_name))))
        if checkpoints:
            #get best checkpoint (with the lowest valid_loss)
            last_checkpoint = checkpoints[-1]
        else:
            logging.warning('no checkpoint found in %s to continue training the model', output_dir)
            #deactivating continue_training for the rest of the script
            continue_training = False
        #if history_file exist, rename to _history.json.old[number]
        history_files = natsorted(glob.glob(os.path.join(output_dir, '{}_history.json'.format(model_name))))
        history_files_old = natsorted(glob.glob(os.path.join(output_dir, '{}_history.json.old*'.format(model_name))))
        if history_files:
            history_file = history_files[-1]
            extracted_number = 1
            if history_files_old:
                file = history_files_old[-1]
                match_old = re.search(r'json.old\d+', file)
                if match_old:
                    extracted_number = int(file[match_old.start()+8:])+1
            old_history_file = '{}.old{}'.format(history_file, extracted_number)
            move(history_file, old_history_file)

    train_keys = scenario['train']
    memmap_datasets = [MemmapDataset(dir=d.get('memmap_dir'), name=d.get('memmap_name')) for d in train_keys]
    train_dataset = reduce(add, memmap_datasets)

    cv = CVSplit(0.2, random_state=0)
    train_dataset, valid_dataset = cv(train_dataset)

    def train_split_func(train_dataset, _):
        return train_dataset, valid_dataset

    skorch_config = {
        'module': model['class_name'],
        'optimizer': torch.optim.RMSprop,
        'optimizer__momentum': 0.9,
        'optimizer__weight_decay': 0.0005,
        'callbacks': [
            # ('auc', skorch.callbacks.EpochScoring(scoring='roc_auc', lower_is_better=False, use_caching=False)),
            ('checkpoints', skorch.callbacks.Checkpoint(os.path.join(output_dir, model_name+'__{last_epoch[epoch]}.pt'))),
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
    }

    skorch_config = {**skorch_config, **model['params']}
    net = NeuralNet(**skorch_config)
    history_file_name = model_name+'_history.json'
    if continue_training and last_checkpoint and old_history_file:
        net.set_params(warm_start=True)
        net.initialize()
        net.load_params(last_checkpoint)
        net.load_history(old_history_file)
        logging.info('net initialized with last checkpoint and history_file')
    net.fit(train_dataset, y=None)
    # net.save_params(os.path.join(output_dir, model_name+'_params'))
    net.save_history(os.path.join(output_dir, model_name+'_history.json'))
