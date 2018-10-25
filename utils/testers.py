import os

import sys
sys.path.append('/dycog/Jeremie/Loic/deepeeg')

import logging
import mne
import shelve
import copy
from datasets.base import P300Dataset
from utils.db import hash_string

import numpy as np
import pandas as pd

class SpellingPredictor(object):
    def __init__(self, nb_items):
        self.nb_items = nb_items

        _priors = None
        self._init()

    def _init(self):
        self._priors = np.ones(self.nb_items)*1/self.nb_items
        self._posts = np.zeros(self.nb_items)

    def reset(self):
        self._init()

    def update(self, target_proba, flashed_item):
        # print('{} / {}'.format(target_proba, flashed_item))
        # # print('{}, {}'.format(target_proba, flashed_item))
        assert target_proba >= 0. and target_proba <= 1.
        for item in range(1, self.nb_items+1):
            i = item - 1
            if flashed_item == item:
                # # print('up: {}'.format(flashed_item))
                self._posts[i] = np.log(self._priors[i]) + np.log(target_proba)
            else:
                # # print('no: {}'.format(flashed_item))
                self._posts[i] = np.log(self._priors[i]) + np.log(0.5)
        self._posts = self._posts - np.max(self._posts) + 1
        self._posts = np.exp(self._posts)
        self._posts=self._posts/np.sum(self._posts)
        self._priors = self._posts

    def get_best_item(self):
        return np.argmax(self._priors) + 1

    def get_priors(self):
        return self._priors
            #
            # self.priors = self.posts
        # posts = np.array([np.log(self.priors[item_index-1]) + np.log(target_proba) for item_index in range(1, self.nb_items+1) if flashed_item == item_index])
        # posts = np.array([np.log(self.priors[item_index-1]) + np.log(0.5) for item_index in range(1, self.nb_items+1) if flashed_item != item_index])
        #

from utils.render import compute_spelling_test_acc

class SpellingTestResult(object):
    def __init__(self):
        self.group_by_item = []
        self.current_item = {}

        self.key_names = {
            'pef': 'priors_every_flash',
            'pei': 'priors_every_repetition',
            'ti': 'true_item',
            'pi': 'predicted_item_repetition'
        }

    def render(self):
        print('rendering')
        test_spelling_result = self.__dict__
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
        acc = compute_spelling_test_acc(group_by_item_df, data_keys=['1_predicted_item_repetition'], true_item_key='true_item', nb_repetitions=nb_repetitions)
        return acc
            # if total_df is None:
            #     total_df = pd.DataFrame(columns=acc_df['after_x_rep'].tolist() + ['name'])
            # total_df.loc[len(total_df),:] = acc_df['acc'].tolist() + [train_test]

    def flush_item(self):
        self.group_by_item.append(copy.copy(self.current_item))
        self.current_item = {}

    def _append_or_create(self, key, obj):
        if key not in self.current_item or not isinstance(self.current_item[key], list):
            self.current_item[key] = []
        self.current_item[key].append(obj)

    def append_priors_on_flash(self, priors):
        # print('append_priors_on_flash')

        key = self.key_names['pef']
        self._append_or_create(key, priors)

    def set_true_item(self, item):
        # print('set_true_item')
        key = self.key_names['ti']
        self.current_item[key] = item

    def append_priors_on_repetition(self, priors):
        # print('append_priors_on_repetition')
        # print(priors)
        key = self.key_names['pei']
        self._append_or_create(key, priors)
        # # print(len(self.current_item[key]))

    def append_predicted_item_on_repetition(self, predicted_item):
        # print('append_predicted_item_on_repetition')
        key = self.key_names['pi']
        self._append_or_create(key, predicted_item)

class P300DatasetTester():
    def __init__(self, classifier, test_data):
        self.test_data = test_data
        self.classifier = classifier
        self.logger = logging.getLogger(__name__)

    def run(self):
        test_data = self.test_data
        X_test_dict = {'x': test_data.X}
        y_pred = self.classifier.predict(X_test_dict)
        y_test = test_data.y

        from sklearn.metrics import confusion_matrix
        confusion_matrix = confusion_matrix(y_test, y_pred)
        test_result = {'confusion_matrix': confusion_matrix}
        return test_result

class P300DatasetTesterSpelling():
    def __init__(self, classifier, test_data, log_proba=True):
        self.test_data = test_data
        self.classifier = classifier
        self.log_proba = log_proba
        self.logger = logging.getLogger(__name__)
        self.sts = SpellingTestResult()

    def run(self):
        test_data = self.test_data
        nb_items = test_data.info['nb_items']
        repetition_length = test_data.info['repetition_length']
        nb_repetitions_by_item = test_data.info['nb_repetitions_by_item']
        vocab_length = test_data.info['vocab_length']
        print('nb_items: {}'.format(nb_items))
        print('repetition_length: {}'.format(vocab_length))
        print('nb_repetitions_by_item: {}'.format(nb_repetitions_by_item))
        print('vocab_length: {}'.format(vocab_length))

        # nb_repetitions = test_data.info['nb_repetitions']
        predictor = SpellingPredictor(vocab_length)

        test_spelling_result = {'group_by_item': []}

        to_index = len(test_data.y_stim) // (repetition_length)

        nb_repetitions_by_item_expanded = [[item for i in range(item)] for item in list(nb_repetitions_by_item)]
        nb_repetitions_by_item_expanded = [item for sublist in nb_repetitions_by_item_expanded for item in sublist]

        assert np.sum(nb_repetitions_by_item) == to_index
        rep_counter = 0

        predictor.reset()
        nb_repetitions = nb_repetitions_by_item_expanded[0]
        for index in range(0, to_index): #len(test_data.y_stim)

            flash_index = int(index*repetition_length)
            # repetition_index = int(index%nb_repetitions)+1

            #every item
            # # print('{} == {} / {}'.format(rep_counter, nb_repetitions, rep_counter == nb_repetitions))
            if rep_counter == nb_repetitions:
                predictor.reset()
                if index != 0:
                    self.sts.flush_item()
                rep_counter = 0
            rep_counter += 1
            nb_repetitions = nb_repetitions_by_item_expanded[index]

            X_one_repetition = test_data.X[flash_index:flash_index+repetition_length]
            y_stim_one_repetition = test_data.y_stim[flash_index:flash_index+repetition_length]
            y_one_repetition = test_data.y[flash_index:flash_index+repetition_length]

            y_pred_one_repetition = self.classifier.predict_proba(X_one_repetition)
            if self.log_proba:
                y_pred_one_repetition = np.exp(y_pred_one_repetition)

            for i in range(len(y_pred_one_repetition)):
                #[i][1] 1 correspond to the target proba
                target_proba = y_pred_one_repetition[i][1]
                #p300 speller protocol (update proba for every symbol in a row/col)
                if isinstance(y_stim_one_repetition[i], np.ndarray):
                    for v in y_stim_one_repetition[i]:
                        predictor.update(target_proba, v)
                #rsvp protocol - simply update the symbol
                else:
                    predictor.update(target_proba, y_stim_one_repetition[i])
                #every flash
                self.sts.append_priors_on_flash(predictor.get_priors())

            #finding the true_item
            item_true = None
            if isinstance(y_stim_one_repetition[0], np.ndarray):
                cross_items = y_stim_one_repetition[np.where(y_one_repetition == 1)]
                assert cross_items.shape[0] == 2
                #p300 speller protocol
                item_true = np.asscalar(np.intersect1d(cross_items[0], cross_items[1]))
            else:
                #rsvp protocol
                item_true = np.asscalar(y_stim_one_repetition[np.where(y_one_repetition == 1)])

            item_predicted_from_prior = predictor.get_best_item()

            # symbol_true = test_data.y_stim_names.get(item_true, None)
            # predicted_symbol = test_data.y_stim_names.get(item_predicted_from_prior, None)

            prior_flash = predictor.get_priors()

            #every repetition
            self.sts.append_priors_on_repetition(predictor.get_priors())
            self.sts.append_predicted_item_on_repetition(predictor.get_best_item())

            self.sts.set_true_item(item_true)

        #dont forget last item
        self.sts.flush_item()
        return self.sts
