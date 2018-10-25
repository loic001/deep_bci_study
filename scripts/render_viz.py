import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from datasets.rsvp import RSVP_COLOR_116
from datasets.meeg import P300_SPELLER_MEEG

mlp__cnn1d_t2 = pickle.load(open('/data/final_results/within-subject/rsvp_meeg_mlp_cnn1d_t2_800/viz.pkl', 'rb'))
# cnn1d_t2_gru = pickle.load(open('/data/final_results/within-subject/rsvp_meeg_cnn1d_t2_gru_800/viz.pkl', 'rb'))
cnn1d_t2_within = [item for item in mlp__cnn1d_t2 if item['model'] == 'cnn1d_t2']
cnn1d_t2_cross_subject = pickle.load(open('/data/final_results/cross-subject/rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru/viz_cnn1d_t2.pkl', 'rb'))


res = cnn1d_t2_within + cnn1d_t2_cross_subject
# res[0]['expe']

plot_dict = [{
    'model': 'cnn1d_t2',
},
# {
#      'model': 'mlp'
# },
# {
#      'model': 'cnn1d_t2_gru'
# }
]

to_plot = []
for item in plot_dict:
    model = item.get('model')
    for expe in ['meeg48', 'rsvp48', 'meeg48_transfer', 'rsvp48_transfer']:
        grad_target = [np.expand_dims(rec['smooth_grad_target'].mean(axis=0), axis=0) for rec in res if (rec['model'] == model and rec['expe'] == expe)]
        grad_non_target = [np.expand_dims(rec['smooth_grad_non_target'].mean(axis=0), axis=0) for rec in res if (rec['model'] == model and rec['expe'] == expe)]
        grad_target_mean = np.concatenate(grad_target, axis=0).mean(axis=0)
        grad_non_target_mean = np.concatenate(grad_non_target, axis=0).mean(axis=0)
        grad_diff_mean = grad_target_mean - grad_non_target_mean
        to_plot.append({
            'grad_target_mean': grad_target_mean,
            'grad_non_target_mean': grad_non_target_mean,
            'grad_diff_mean': grad_diff_mean,
            'expe': expe,
            'model': model
        })

exclude_channels = ['P9', 'P10', 'PO9', 'PO3', 'PO4', 'PO10', 'Oz', 'Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FT7', 'FT8']
raw_vpfat = RSVP_COLOR_116().get_subject_data('VPfat')
raw_vpfat.pick_types(eeg=True, exclude=exclude_channels)
raw_vpfat.info['sfreq'] = 100.0
print(raw_vpfat.info['ch_names'])

# raw_s02 = P300_SPELLER_MEEG().get_subject_data('S02')
# raw_s02.pick_types(eeg=True, exclude=exclude_channels)
# raw_s02.info['sfreq'] = 100.0
# print(raw_s02.info['ch_names'])

show=False
figs = []
figs_joint = []
for item in to_plot:
    grad_diff_mean = item.get('grad_diff_mean')
    expe = item.get('expe')
    model = item.get('model')
    evoked_target = mne.EvokedArray(grad_diff_mean, raw_vpfat.info)
    scalings = {'eeg': 10**9} if expe == 'rsvp48' else {'eeg': 10**7}
    titles = {'eeg': '{}_{}'.format(expe, model)}
    fig = evoked_target.plot_image(units={'eeg': 'Gradient'}, titles=titles, scalings=scalings, show=show)
    fig_joint = evoked_target.plot_joint(show=show, title=titles['eeg'])
    figs.append(fig)
    figs_joint.append(fig_joint)
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

multipage('viz_cnn1d_t2.pdf', figs + figs_joint)
