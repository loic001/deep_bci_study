import numpy as np
import pandas as pd

def compute_spelling_test_acc(group_by_item_df, data_keys, true_item_key, nb_repetitions):
    data = group_by_item_df
    accuracies = []
    print(data[['predicted_item_repetition', 'true_item']])
    for x_iter in range(nb_repetitions):
        boolean_or = []
        for data_key in data_keys:
            #new
            union_bool = data[data_key].apply(lambda x: x[x_iter]) == data[true_item_key]
            boolean_or.append(union_bool)
        res_df = boolean_or[0]
        for b in boolean_or[1:]:
            res_df = res_df | b
        print(len(res_df))
        acc = np.mean(res_df)
        accuracies.append({'after_x_rep': x_iter + 1, 'acc': acc})
    return accuracies

def build_spelling_test_figure(name, test_spelling_result, fontsize):
    fig = plt.figure()
    fig.suptitle(name, fontsize=fontsize)
    test_spelling_grid = gridspec.GridSpec(2, 2)

    acc_iter_ax = fig.add_subplot(test_spelling_grid[0, 0])
    acc_iter_ax_2 = fig.add_subplot(test_spelling_grid[0, 1])
    grouped_by_item = test_spelling_result['group_by_item']
    group_by_item_df = pd.DataFrame(grouped_by_item)
    nb_items = len(group_by_item_df['priors_every_repetition'][0][0])
    for i in range(1, nb_items+1):
        key = '{}_predicted_item_repetition'.format(i)
        group_by_item_df[key] = group_by_item_df['priors_every_repetition'].apply(lambda x: [np.argsort(p, axis=0)[-i]+1 for p in x])


    nb_repetitions = np.min(np.unique(np.array([len(p) for p in group_by_item_df['priors_every_repetition']])))
    # nb_repetitions = len(group_by_item_df['priors_every_repetition'][0])

    for k in range(nb_repetitions):
        key_current = 'predicted_item_sorted_repetition_{}'.format(k+1)
        group_by_item_df[key_current] = group_by_item_df.index.tolist()
        group_by_item_df[key_current] = group_by_item_df[key_current].apply(lambda x: [group_by_item_df['{}_predicted_item_repetition'.format(i)][x][k] for i in range(1, nb_items+1)])

    # generate html table
    # keys = ['true_item']
    # keys += ['{}_predicted_item_repetition'.format(i+1) for i in range(nb_items)]
    # keys += ['predicted_item_sorted_repetition_{}'.format(i+1) for i in range(nb_repetitions)]
    # to_generate_df = group_by_item_df[keys]
    # symb = ['a']
    # import string
    # letters = list(string.ascii_lowercase)
    # letters.append('_')
    # letters.append('_')
    # letters.append('_')
    # letters.append('_')
    # letters.append('_')
    # letters.append('_')
    # to_generate_df['letters'] = to_generate_df['true_item'].apply(lambda x: letters[x-1])
    # to_generate_df.to_html('out.html')

    acc_arr = compute_spelling_test_acc(group_by_item_df, data_keys=['1_predicted_item_repetition'], true_item_key='true_item', nb_repetitions=nb_repetitions)
    bar_plot_arr(acc_arr, 'Acc after N repetitions (1 item)', x_key='after_x_rep', y_key='acc', ax=acc_iter_ax, fontsize=fontsize)
    acc_arr_2 = compute_spelling_test_acc(group_by_item_df, data_keys=['1_predicted_item_repetition', '2_predicted_item_repetition'], true_item_key='true_item', nb_repetitions=nb_repetitions)
    bar_plot_arr(acc_arr_2, 'Acc after N repetitions (2 best items)', x_key='after_x_rep', y_key='acc', ax=acc_iter_ax_2, fontsize=fontsize)

    fig.subplots_adjust(wspace=0.7, hspace=0.8)
    fig.tight_layout(pad=5)
    return fig, acc_arr
