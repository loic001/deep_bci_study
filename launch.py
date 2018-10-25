import time
import os
import subprocess
import argparse
import re

from deep_models_def import deep_models_2d
from deep_models_def import deep_models_3d
from riemann_models_def import riemann_models

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', dest='cluster', action='store_true')
parser.add_argument('--no-cluster', dest='cluster', action='store_false')
parser.add_argument('--execute', dest='execute', action='store_true')
parser.add_argument("--output-dir-base", help='output dir', type=str, default='outputs')
parser.add_argument('--seq', dest='seq', action='store_true')
parser.add_argument('--continue', dest='continue_training', action='store_true')


parser.set_defaults(cluster=True, exec=False, seq=False, continue_training=False)

# afterany

args = parser.parse_args()
execute = args.execute
cluster = args.cluster
seq = args.seq

cluster_scenario_dir = '/mnt/data/loic/scenarios'
scenario_dir = '/data/scenarios'

if cluster:
    scenario_dir = cluster_scenario_dir

#--exclude=node2
sbatch_cmd = 'sbatch -J {job_name} -o outputs/log/%j-%x.out -e outputs/log/%j-%x.err -c20 --exclude=node2 --wrap "{cmd}"'
deep_cmd = "python deep_run.py --scenario {scenario_file} --model {model_name}"
deep_cmd = deep_cmd + ' --continue' if args.continue_training else deep_cmd
riemann_cmd = "python riemann_run.py --scenario {scenario_file} --model {model_name} --output-dir-base {output_dir_base}"

_models = ['cnn1d_t2']
_scenarios = ['meeg48_expe_transfer_rsvp48']
_subjects = ['S02', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
        'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S19',
        'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28']

#--------------------------------------------------------------------

meeg_subjects = ['S02', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
        'S11', 'S13', 'S14', 'S15', 'S16', 'S17', 'S19',
        'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28']

rsvp_subjects = ['VPgcc', 'VPfat', 'VPgcb', 'VPgcg', 'VPgcd', 'VPgcf', 'VPgch',
        'VPiay', 'VPicn', 'VPicr', 'VPpia']

default_max_epoch = 400

from itertools import permutations

scenario_names = ['', 'transfer', 'cum', 'expe_transfer']
datasets = {
    'rsvp': rsvp_subjects,
    'meeg': meeg_subjects,
    'rsvp48': rsvp_subjects,
    'meeg48': meeg_subjects
}
formats = ['', 'topo']
scenarios = {}

for scenario_name in scenario_names:
    if scenario_name in ['cum', 'expe_transfer']:
        for (d1, subjets_1), (d2, subjects_2) in permutations(datasets.items(), 2):
            for format in formats:
                models = deep_models_2d + riemann_models if format == '' else deep_models_3d
                format = '_{}'.format(format) if format else ''
                _scenario_name = '_{}'.format(scenario_name) if scenario_name else ''
                scenario = '{}{}{}_{}'.format(d1, format, _scenario_name, d2)
                scenarios[scenario] = {
                    'models': models,
                    'subjects': subjets_1
                }
    else:
        for dataset, subjets in datasets.items():
            for format in formats:
                models = deep_models_2d + riemann_models if format == '' else deep_models_3d
                format = '_{}'.format(format) if format else ''
                _scenario_name = '_{}'.format(scenario_name) if scenario_name else ''
                scenario = '{}{}{}'.format(dataset, format, _scenario_name)
                scenarios[scenario] = {
                    'models': models,
                    'subjects': subjets
                }


scenarios['rsvp48_expe_transfer_meeg48']
# print(scenarios['rsvp48_topo_transfer'])

if '*' in _models:
    _models += deep_models + riemann_models

if '*riemann' in _models:
    _models += riemann_models

if '*deep' in _models:
    _models += deep_models

if '*' in _subjects:
    _subjects += rsvp_subjects + meeg_subjects

if '*rsvp' in _subjects:
    _subjects += rsvp_subjects

if '*meeg' in _subjects:
    _subjects += meeg_subjects

if '*' in _scenarios:
    _scenarios += list(scenarios.keys())

_models = list(set(_models))
_subjects = list(set(_subjects))
_scenarios = list(set(_scenarios))

sbatch_cmds = []
cmds = []
for scenario_key, item in scenarios.items():
    for model in item['models']:
        for subject in item['subjects']:
            scenario_file = os.path.join(scenario_dir, scenario_key, '{}.json'.format(subject))
            expe_cmd = riemann_cmd if model in riemann_models else deep_cmd
            cmd = expe_cmd.format(**{'scenario_file': scenario_file, 'model_name': model, 'output_dir_base': args.output_dir_base})
            job_name = '{}_{}_{}'.format(scenario_key, model, subject)
            final_cmd = sbatch_cmd.format(**{'job_name': job_name, 'cmd': cmd})
            if model in _models and scenario_key in _scenarios and subject in _subjects:
                sbatch_cmds.append(final_cmd)
                cmds.append(cmd)

final_cmds = sbatch_cmds if cluster else cmds
last_submitted_id = None
if not execute:
    for cmd in final_cmds:
        print(cmd)
else:
    for index, cmd in enumerate(final_cmds):
        # os.system(cmd)
        if seq and last_submitted_id is not None:
            cmd += ' --dependency=afterany:{}'.format(last_submitted_id)
        res_bytes = subprocess.check_output(cmd, shell=True)
        res = res_bytes.decode("utf-8")
        submitted_id = None
        try:
            submitted_id = int(re.findall("[0-9]+",res)[-1])
        except:
            print('error submitting job')
        print(submitted_id)
        last_submitted_id = submitted_id

        time.sleep(0.3)
