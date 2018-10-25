import logging
from collections import Counter
import numpy as np
from scipy import interpolate
from scipy.stats import stats
from functools import partial

class Transformer(object):
    def __init__(self):
        self.logger = logging.getLogger(__class__.__name__)

    def transform(self, X, y, **kwargs):
        self.logger.info('X data, shape: {}'.format(X.shape))
        X, y = self._transform(X, y, **kwargs)
        self.logger.info('X transformed data, shape: {}'.format(X.shape))
        return X, y

    def _transform(self, X, y, **kwargs):
        return X, y


class ZScoreTransformer(Transformer):
    def __init__(self):
        super().__init__()

    def _transform(self, X, y, **kwargs):
        return stats.zscore(X, axis=2), y

def interpolate_(ch_names, ch_pos_map, values, size=16):
    assert len(values) == len(ch_names), "Should be as many values as channels"
    vmin, vmax = -np.max(np.abs(values)), np.max(np.abs(values))
    # what if we have an unknown channel?
    data_points = [ch_pos_map[c] for c in ch_names]
    z = [values[i] for i in range(len(data_points))]
    # calculate the interpolation
    x = [i[0] for i in data_points]
    y = [i[1] for i in data_points]
    # interpolate the in-between values
    xx = np.linspace(min(x), max(x), size+4)
    yy = np.linspace(min(y), max(y), size+4)
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    f = interpolate.CloughTocher2DInterpolator(data_points, z, fill_value=10e-10)
    zz = f(xx_grid, yy_grid)
    return zz[2:size+2,2:size+2]

def interpolate_one_epoch_(params):
    epoch, ch_names, ch_pos_map, index_epoch = params
    print('epoch : {}'.format(index_epoch))
    res = np.apply_along_axis(partial(interpolate_, ch_names, ch_pos_map), 0, epoch)
    return res
    #res is shape (x,y,T)

class InterpolatorTransformer(Transformer):
    def __init__(self):
        super().__init__()

    def _transform(self, X, y, **kwargs):
        import multiprocessing
        # from utils.handythread import parallel_map
        processes = kwargs.get('processes', 20)
        pool = multiprocessing.Pool(processes=processes)
        ch_names = kwargs.get('ch_names', None)
        ch_pos_map = kwargs.get('ch_pos_map', None)
        assert ch_names is not None
        assert ch_pos_map is not None
        to_run = []
        for index_epoch, epoch in enumerate(X):
            to_run.append((epoch, ch_names, ch_pos_map, index_epoch))

        interpolated_epochs = pool.map(interpolate_one_epoch_, to_run)
        X_transformed = np.array(interpolated_epochs)
        return X_transformed, y


class SamplingTransformer(Transformer):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def _transform(self, X, y):
        print(X.shape)
        first_dim, second_dim, third_dim = X.shape
        X_reshaped = X.reshape(first_dim, second_dim * third_dim)
        sampled_res = self.sampler.fit_sample(X_reshaped, y)
        X_resampled, y_resampled, *selected_index = sampled_res
        selected_index = selected_index[0] if selected_index else None
        X = X_resampled.reshape(X_resampled.shape[0], second_dim, third_dim)
        y = y_resampled
        return X, y, selected_index
