import json
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams.update(plt.rcParamsDefault)
# plt.style.use(u'ggplot')

sns.set(style="whitegrid", color_codes=True)
sns.set(rc={"figure.figsize": (12, 4)})

np.random.seed(sum(map(ord, "palettes")))
sns.palplot(sns.color_palette("Set2", 8))
sns.set(font_scale=1.5)

def patch_class(df):
    model_class = {
        'riemann_tang_log': 'riemann',
        'riemann_xdawn_tang_log': 'riemann',
        'mlp': 'classic',
        'cnn1d_t2': 'classic',
        'cnn1d_t2_gru': 'classic',
        'cnn2d_s3': 'topo',
        'cnn3d_s3_t2': 'topo',
        'cnn3d_s3_t2_gru': 'topo'
    }
    df['class'] = df['model'].apply(lambda model: model_class[model])

def patch_dataset(df):
    def to_dataset(string):
        if string.find('rsvp') != -1:
            return 'rsvp'
        else:
            return 'meeg'
    df['dataset'] = df['expe'].apply(to_dataset)


def patch_scenario(df):
    def to_scenario(string):
        if string.find('rsvp48_expe_transfer') != -1:
            return 'RSVP only'
        elif string.find('meeg48_expe_transfer') != -1:
            return 'MEEG only'
        elif string.find('expe_transfer') != -1:
            return 'expe_transfer'
        elif string.find('transfer') != -1:
            return 'Cross-subject'
        elif string.find('cum') != -1:
            return 'Cross-experiment'
        else:
            return 'Within-subject'
    df['scenario'] = df['expe'].apply(to_scenario)


def patch_chance_level(df):
    def to_chance_level(string):
        if string.find('rsvp') != -1:
            return 1/30
        else:
            return 1/36
    df['chance_level'] = df['expe'].apply(to_chance_level)



riemann_within = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_riemann/acc.pkl')

rsvp_meeg_mlp_cnn1d_t2_800 = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_mlp_cnn1d_t2_800/acc.pkl')
rsvp_meeg_cnn1d_t2_gru_800 = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn1d_t2_gru_800/acc.pkl')
rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500 = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500/acc.pkl')
rsvp_meeg_cnn2d_s3 = rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500[~(rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500['model'] == 'cnn3d_s3_t2')]
rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000 = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000/acc.pkl')
rsvp_meeg_cnn3d_s3_t2_600 = rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000[~(rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000['model'] == 'cnn3d_s3_t2_gru')]
rsvp_meeg_cnn3d_s3_t2_gru_1700 = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn3d_s3_t2_gru_1700/acc.pkl')

deep_within = pd.concat([rsvp_meeg_mlp_cnn1d_t2_800,
                         rsvp_meeg_cnn1d_t2_gru_800,
                         rsvp_meeg_cnn2d_s3,
                         rsvp_meeg_cnn3d_s3_t2_600,
                         rsvp_meeg_cnn3d_s3_t2_gru_1700])

rsvp_meeg_mlp_cnn1d_t2_800_vl = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_mlp_cnn1d_t2_800/valid_loss.pkl')
rsvp_meeg_cnn1d_t2_gru_800_vl = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn1d_t2_gru_800/valid_loss.pkl')
rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500_vl = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500/valid_loss.pkl')
rsvp_meeg_cnn2d_s3_vl = rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500_vl[~(rsvp_meeg_cnn2d_s3_cnn3d_s3_t2_500_vl['model'] == 'cnn3d_s3_t2')]
rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000_vl = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000/valid_loss.pkl')
rsvp_meeg_cnn3d_s3_t2_600_vl = rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000_vl[~(rsvp_meeg_cnn3d_s3_t2_600_cnn3d_s3_t2_gru_1000_vl['model'] == 'cnn3d_s3_t2_gru')]
rsvp_meeg_cnn3d_s3_t2_gru_1700_vl = pd.read_pickle('/data/final_results/within-subject/rsvp_meeg_cnn3d_s3_t2_gru_1700/valid_loss.pkl')

deep_within_vl = pd.concat([rsvp_meeg_mlp_cnn1d_t2_800_vl,
                         rsvp_meeg_cnn1d_t2_gru_800_vl,
                         rsvp_meeg_cnn2d_s3_vl,
                         rsvp_meeg_cnn3d_s3_t2_600_vl,
                         rsvp_meeg_cnn3d_s3_t2_gru_1700_vl])

#cross

rsvp_meeg_riemann_xdawn_tang_log_cross = pd.read_pickle('/data/final_results/cross-subject/rsvp_meeg_riemann_xdawn_tang_log/acc.pkl')
rsvp_riemann_tang_log_cross = pd.read_pickle('/data/final_results/cross-subject/rsvp_meeg_riemann_tang_log/acc.pkl')
riemann_cross = pd.concat([rsvp_meeg_riemann_xdawn_tang_log_cross,
                           rsvp_riemann_tang_log_cross])

rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru_cross = pd.read_pickle('/data/final_results/cross-subject/rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru/acc.pkl')
rsvp_meeg_cnn2d_s3_1000_cross = pd.read_pickle('/data/final_results/cross-subject/rsvp_meeg_cnn2d_s3_1000/acc.pkl')
rsvp_cnn3d_s3_t2_1000 = pd.read_pickle('/data/final_results/cross-subject/rsvp_cnn3d_s3_t2_1000/acc.pkl')
meeg_cnn3d_s3_t2_1000 = pd.read_pickle('/data/final_results/cross-subject/meeg_cnn3d_s3_t2_1000/acc.pkl')
# rsvp_cnn3d_s3_t2_gru_1700 = pd.read_pickle('/data/final_results/cross-subject/rsvp_cnn3d_s3_t2_gru_1700/acc.pkl')
rsvp_cnn3d_s3_t2_gru_1900 = pd.read_pickle('/data/final_results/cross-subject/rsvp_cnn3d_s3_t2_gru_1900/acc.pkl')
# meeg_cnn3d_s3_t2_gru_1700 = pd.read_pickle('/data/final_results/cross-subject/meeg_cnn3d_s3_t2_gru_1700/acc.pkl')
meeg_cnn3d_s3_t2_gru_1900 = pd.read_pickle('/data/final_results/cross-subject/meeg_cnn3d_s3_t2_gru_1900/acc.pkl')

deep_cross = pd.concat([rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru_cross,
                        rsvp_meeg_cnn2d_s3_1000_cross,
                        rsvp_cnn3d_s3_t2_1000,
                        meeg_cnn3d_s3_t2_1000,
                        rsvp_cnn3d_s3_t2_gru_1700,
                        meeg_cnn3d_s3_t2_gru_1700])


rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru_cross_vl = pd.read_pickle('/data/final_results/cross-subject/rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru/valid_loss.pkl')
rsvp_meeg_cnn2d_s3_1000_cross_vl = pd.read_pickle('/data/final_results/cross-subject/rsvp_meeg_cnn2d_s3_1000/valid_loss.pkl')
rsvp_cnn3d_s3_t2_1000_vl = pd.read_pickle('/data/final_results/cross-subject/rsvp_cnn3d_s3_t2_1000/valid_loss.pkl')
meeg_cnn3d_s3_t2_1000_vl = pd.read_pickle('/data/final_results/cross-subject/meeg_cnn3d_s3_t2_1000/valid_loss.pkl')
# rsvp_cnn3d_s3_t2_gru_1700 = pd.read_pickle('/data/final_results/cross-subject/rsvp_cnn3d_s3_t2_gru_1700/acc.pkl')
rsvp_cnn3d_s3_t2_gru_1900_vl = pd.read_pickle('/data/final_results/cross-subject/rsvp_cnn3d_s3_t2_gru_1900/valid_loss.pkl')
# meeg_cnn3d_s3_t2_gru_1700 = pd.read_pickle('/data/final_results/cross-subject/meeg_cnn3d_s3_t2_gru_1700/acc.pkl')
meeg_cnn3d_s3_t2_gru_1900_vl = pd.read_pickle('/data/final_results/cross-subject/meeg_cnn3d_s3_t2_gru_1900/valid_loss.pkl')

deep_cross_vl = pd.concat([rsvp_meeg_mlp_cnn1d_t2_cnn1d_t2_gru_cross_vl,
                        rsvp_meeg_cnn2d_s3_1000_cross_vl,
                        rsvp_cnn3d_s3_t2_1000_vl,
                        meeg_cnn3d_s3_t2_1000_vl,
                        rsvp_cnn3d_s3_t2_gru_1900_vl,
                        meeg_cnn3d_s3_t2_gru_1900_vl])


rsvp48_cum_meeg48_cnn1d_t2_cnn3d_s3_t2 = pd.read_pickle('/data/final_results/cum/rsvp48_cum_meeg48_cnn1d_t2_cnn3d_s3_t2/acc.pkl')


rsvp48_exp_transfer_meeg48 = pd.read_pickle('/dycog/Jeremie/Loic/v2/outputs/exp_transfer/acc.pkl')
meeg48_exp_transfer_rsvp48 = pd.read_pickle('/dycog/Jeremie/Loic/v2/outputs/exp_transfer_2/acc.pkl')


acc = pd.concat([deep_within, riemann_within, deep_cross, riemann_cross, rsvp48_cum_meeg48_cnn1d_t2_cnn3d_s3_t2, rsvp48_exp_transfer_meeg48, meeg48_exp_transfer_rsvp48])
patch_class(acc)
patch_scenario(acc)
patch_dataset(acc)
patch_chance_level(acc)

vl = pd.concat([deep_within_vl, deep_cross_vl])
patch_class(vl)

meeg_within = acc.loc[(acc['expe'].isin(['meeg48', 'meeg48_topo'])) & (acc['rep'] == 2)]
meeg_transfer = acc.loc[(acc['expe'].isin(['meeg48_transfer', 'meeg48_topo_transfer'])) & (acc['rep'] == 2)]

rsvp_within = acc.loc[(acc['expe'].isin(['rsvp48', 'rsvp48_topo'])) & (acc['rep'] == 2)]# & ~(acc['model'] == 'cnn3d_s_t')
rsvp_transfer = acc.loc[(acc['expe'].isin(['rsvp48_transfer', 'rsvp48_transfer', 'rsvp48_topo_transfer'])) & (acc['rep'] == 2)]

rsvp_cum_meeg = acc.loc[(acc['expe'].isin(['rsvp48_cum_meeg48', 'rsvp48_topo_cum_meeg48'])) & (acc['rep'] == 2)]
rsvp_expe_transfer_meeg = acc.loc[(acc['expe'].isin(['rsvp48_expe_transfer_meeg48', 'rsvp48_topo_expe_transfer_meeg48'])) & (acc['rep'] == 2)]
meeg_expe_transfer_rsvp = acc.loc[(acc['expe'].isin(['meeg48_expe_transfer_rsvp48', 'meeg48_topo_expe_transfer_rsvp48'])) & (acc['rep'] == 2)]

rsvp_cum_meeg_plus_rsvp_transfer = pd.concat([rsvp_cum_meeg, rsvp_transfer[rsvp_transfer['model'].isin(['cnn1d_t2'])], rsvp_expe_transfer_meeg, meeg_expe_transfer_rsvp])


#export r dataframe
# acc=acc[acc['scenario'].isin(['Cross-subject', 'Within-subject'])]
# rsvp = acc[(acc['dataset'] == 'rsvp')]
# meeg = acc[(acc['dataset'] == 'meeg')]
#
# rsvp['acc_norm'] = rsvp['acc']/rsvp['chance_level']
#
# rsvp['acc_norm'] = rsvp['acc']/rsvp['chance_level']
# meeg['acc_norm'] = meeg['acc']/meeg['chance_level']
# rsvp['acc_norm_log'] = np.log(rsvp['acc']/rsvp['chance_level'])
# meeg['acc_norm_log'] = np.log(meeg['acc']/meeg['chance_level'])
# rsvp['acc_norm_diff'] = rsvp['acc']/rsvp['chance_level']
# meeg['acc_norm_diff'] = meeg['acc']/meeg['chance_level']
#
#
# meeg['acc_norm'] = np.log(meeg['acc']/meeg['chance_level'])
# rsvp_meeg_norm = pd.concat([rsvp, meeg])
# import feather
# feather.write_dataframe(rsvp_meeg_norm, '/home/loic.delobel/loic_models_norm.feather')

#gen err table
# vl['gen_err'] = vl['valid_loss'] - vl['train_loss']
# meeg_within_vl = vl.loc[(vl['expe'].isin(['meeg48', 'meeg48_topo']))]
# meeg_transfer_vl = vl.loc[(vl['expe'].isin(['meeg48_transfer', 'meeg48_topo_transfer']))]
#
# rsvp_within_vl = vl.loc[(vl['expe'].isin(['rsvp48', 'rsvp48_topo']))]
# rsvp_transfer_vl = vl.loc[(vl['expe'].isin(['rsvp48_transfer', 'rsvp48_transfer', 'rsvp48_topo_transfer']))]
#
#
# meeg_within_vl_grouped_model = meeg_within_vl.groupby('model').apply(lambda values: values['gen_err'].mean())
#
# meeg_transfer_vl_grouped_model = meeg_transfer_vl.groupby('model').apply(lambda values: values['gen_err'].mean())
# rsvp_within_vl_grouped_model = rsvp_within_vl.groupby('model').apply(lambda values: values['gen_err'].mean())
# rsvp_transfer_vl_grouped_model = rsvp_transfer_vl.groupby('model').apply(lambda values: values['gen_err'].mean())
#
# gen_err_table = pd.concat([meeg_within_vl_grouped_model, meeg_transfer_vl_grouped_model, rsvp_within_vl_grouped_model, rsvp_transfer_vl_grouped_model],axis=1)
# gen_err_table.columns = ['meeg_within','meeg_transfer','rsvp_within','rsvp_transfer']
# json_gen_table = gen_err_table.to_json(orient='columns')
# with open('gen_err.json', 'w') as f:
#     f.write(json_gen_table)

def cat_plot(data, x_key='model', y_key='acc', hue_key='class', order=None, ax=None, chance_acc=None, title=None, dodge=False, x_labels=None):
    data=data.sort_values(x_key)
    sns.catplot(x=x_key, y=y_key, hue=hue_key, kind="bar", order=order, data=data, ax=ax, dodge=dodge)
    # sns.lineplot(data=pd.DataFrame([0.2, 0.2, 0.2, 0.2, 0.2]), dashes=True, ax=ax, color='red')
    if chance_acc:
        ax.axhline(y=chance_acc, linewidth=2, color='r', ls='--', label="chance level")
    ax.set(xlabel='', ylabel='Mean accuracy')
    ax.legend(loc=1, prop={'size': 14})

    ax.set_title(title)
    if x_labels:
        ax.set_xticklabels(x_labels)
    return ax

y_key = 'acc'
scenarios = [
    # {
    #     'title': 'Within-subject',
    #     'dodge': False,
    #     'hue_key': 'class',
    #     'plot_order': ['riemann_tang_log', 'riemann_xdawn_tang_log', 'mlp', 'cnn1d_t2', 'cnn1d_t2_gru', 'cnn2d_s3', 'cnn3d_s3_t2', 'cnn3d_s3_t2_gru'],
    #     'labels': ['Riemann', 'XDAWN + Riemann', 'MLP', 'CNN1D_T2', 'CNN1D_T2_GRU', 'CNN2D_S3', 'CNN3D_S3_T2', 'CNN3D_S3_T2_GRU'],
    #     'data': [
    #         {
    #             'name': 'RSVP',
    #             'data': rsvp_within,
    #             'chance_acc': 1/30
    #         },
    #         {
    #             'name': 'MEEG',
    #             'data': meeg_within,
    #             'chance_acc': 1/36
    #         }
    #     ]
    # },
    # {
    #     'title': 'Cross-subject',
    #     'dodge': False,
    #     'hue_key': 'class',
    #     'plot_order': ['riemann_tang_log', 'riemann_xdawn_tang_log', 'mlp', 'cnn1d_t2', 'cnn1d_t2_gru', 'cnn2d_s3', 'cnn3d_s3_t2', 'cnn3d_s3_t2_gru'],
    #     'labels': ['Riemann', 'XDAWN + Riemann', 'MLP', 'CNN1D_T2', 'CNN1D_T2_GRU', 'CNN2D_S3', 'CNN3D_S3_T2', 'CNN3D_S3_T2_GRU'],
    #     'data': [
    #         {
    #             'name': 'RSVP',
    #             'data': rsvp_transfer,
    #             'chance_acc': 1/30
    #         },
    #         {
    #             'name': 'MEEG',
    #             'data': meeg_transfer,
    #             'chance_acc': 1/36
    #         }
    #     ]
    # },
        {
            'title': 'Cross-experiment RSVP + MEEG',
            'plot_order': ['cnn1d_t2'],
            'labels': ['CNN1D_T2'],
            'hue_key': 'scenario',
            'dodge': True,
            'data': [
                {
                    'name': 'RSVP_MEEG',
                    'data': rsvp_cum_meeg_plus_rsvp_transfer,
                    'chance_acc': 1/30
                }
            ]
        }
]
# plot_order=plot_order_vl
# y_key = 'valid_loss'
# scenarios = [
#     {
#         'title': 'Within-subject',
#         'data': [
#             {
#                 'name': 'RSVP Best valid loss',
#                 'data': rsvp_within_vl,
#             },
#             {
#                 'name': 'MEEG Best valid loss',
#                 'data': meeg_within_vl,
#             }
#         ]
#     },
#     # {
#     #     'title': 'Cross-subject',
#     #     'data': [
#     #         {
#     #             'name': 'RSVP Best valid loss',
#     #             'data': rsvp_transfer_vl,
#     #             'chance_acc': 1/30
#     #         },
#     #         {
#     #             'name': 'MEEG Best valid loss',
#     #             'data': meeg_transfer_vl,
#     #             'chance_acc': 1/36
#     #         }
#     #     ]
#     # }
# ]
# fig = plt.figure()
# fig.suptitle('ti', fontsize=10)
# grid = gridspec.GridSpec(3, 1)
# ax1 = fig.add_subplot(grid[:3, 0])
# cat_plot(rsvp_within, title='_data['name']', ax=ax1)
#
pvalue_th = 0.05
figs = []
for index_scenario, scenario in enumerate(scenarios):
    title = scenario['title']
    plot_order = scenario['plot_order']
    labels = scenario['labels']
    dodge = scenario['dodge']
    hue_key = scenario['hue_key']
    for index, _data in enumerate(scenario['data']):
        font_size = 10
        fig = plt.figure()
        # fig.suptitle(title, fontsize=font_size)
        grid = gridspec.GridSpec(3, 1)
        data = _data['data']
        ax1 = fig.add_subplot(grid[:3, 0])
        # ax2 = fig.add_subplot(grid[2, 0])
        cat_plot(data, title='{} {}'.format(title, _data['name']), ax=ax1, hue_key=hue_key, y_key=y_key, order=plot_order, chance_acc=_data.get('chance_acc'), dodge=dodge, x_labels=labels)
        # sns.catplot(x='model', y='acc', hue='class', kind="bar", data=data)
        # _data_test = wilcoxon_test_all(data)
        # _data_test_pvalue_matrix = pd.DataFrame(_data_test).pivot('ref_model', 'model', 'pvalue')
        # _data_test_pvalue_matrix = _data_test_pvalue_matrix.applymap(lambda x: x*6)
        # print(_data_test_pvalue_matrix)
        # _data_test_sign_matrix = _data_test_pvalue_matrix.applymap(partial(threshold, pvalue_th=pvalue_th))
        # sns.heatmap(_data_test_sign_matrix, square=True, cmap='Blues', cbar=False, ax=ax2)
        # ax2.set_title('significance level: {}'.format(pvalue_th))
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.tight_layout()
    figs.append(fig)
#
# fig.show()

plt.show()
