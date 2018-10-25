import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from natsort import natsorted

mlp__cnn1d_t2 = pickle.load(open('/data/final_results/within-subject/rsvp_meeg_mlp_cnn1d_t2_800/viz.pkl', 'rb'))
# cnn1d_t2_gru = pickle.load(open('/data/final_results/within-subject/rsvp_meeg_cnn1d_t2_gru_800/viz.pkl', 'rb'))
cnn1d_t2_within = [item for item in mlp__cnn1d_t2 if item['model'] == 'cnn1d_t2']
cnn1d_t2_cross_subject = pickle.load(open('/data/final_results/cross-subject/rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru/viz_cnn1d_t2.pkl', 'rb'))

res = cnn1d_t2_within + cnn1d_t2_cross_subject


rsvp_within = [item for item in res if item['expe'] == 'rsvp48']
meeg_within = [item for item in res if item['expe'] == 'meeg48']
rsvp_cross = [item for item in res if item['expe'] == 'rsvp48_transfer']
meeg_cross = [item for item in res if item['expe'] == 'meeg48_transfer']

images = {}

subjects_rsvp = natsorted([item['subject'] for item in rsvp_within])
subjects_meeg = natsorted([item['subject'] for item in meeg_within])

obs = []
for item in res:
    grad_target_mean = item['smooth_grad_target'].mean(axis=0)
    grad_non_target_mean = item['smooth_grad_non_target'].mean(axis=0)
    grad_shape = grad_target_mean.shape
    for dim1_index in range(grad_shape[0]):
        for dim2_index in range(grad_shape[1]):
            new_item = {}
            new_item['expe'] = item['expe']
            new_item['model'] = item['model']
            new_item['subject'] = item['subject']
            new_item['coord_string'] = '{}_{}'.format(dim1_index, dim2_index)
            new_item['coord_dim1'] = dim1_index
            new_item['coord_dim2']  = dim2_index
            new_item['gradient_target'] = grad_target_mean[dim1_index, dim2_index]
            new_item['gradient_non_target'] = grad_non_target_mean[dim1_index, dim2_index]
            obs.append(new_item)

df=pd.DataFrame(obs)
import multiprocessing as mp
from scipy.stats import ttest_1samp

def get_stat_image(df, key='gradient_target'):
    def test(values):
        return ttest_1samp(values, 0.0)
    def to_image(stats):
        grad_shape = (48, 61)
        image = np.ones(grad_shape)
        for index, coord_string in enumerate(stats.index):
            cut = coord_string.find('_')
            dim1_index = int(coord_string[:cut])
            dim2_index = int(coord_string[cut+1:])
            image[dim1_index, dim2_index] = stats.iloc[index]
        return image
    stats = df.groupby('coord_string').apply(lambda v: v[key].as_matrix()).apply(test).apply(lambda x: x[1])
    image = to_image(stats)
    return image

df_rsvp48 = df[df['expe'] == 'rsvp48']
df_rsvp48_transfer = df[df['expe'] == 'rsvp48_transfer']
df_meeg48 = df[df['expe'] == 'meeg48']
df_meeg48_transfer = df[df['expe'] == 'meeg48_transfer']

rsvp48_image = get_stat_image(df_rsvp48)
rsvp48_transfer_image = get_stat_image(df_rsvp48_transfer)
meeg48_image = get_stat_image(df_meeg48)
meeg48_transfer_image = get_stat_image(df_meeg48_transfer)

plt.imshow(rsvp48_image<0.01)
0.01/(48*61)
plt.imshow(rsvp48_image<0.01/(48*61), origin='lower')


from statsmodels.stats import multitest

rsvp48_image.reshape(-1)



reject, p_vals, _, _ = multitest.multipletests(rsvp48_image.reshape(-1), alpha=0.05, method='holm')

p_vals

image = np.vstack([np.expand_dims(item['smooth_grad_target'].mean(axis=0), axis=0) for item in res if item['expe'] == 'meeg48_transfer']).mean(axis=0)


skimage.measure.block_reduce(rsvp48_image, (1,3), np.min).shape

plt.imshow(rsvp48_image<=0.1, origin='lower')

plt.imshow(np.abs(image)>0.000000020, origin='lower')
