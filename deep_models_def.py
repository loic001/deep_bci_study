from torch_ext.datasets import MemmapDataset

from models.mlp import MLP
from models.cnn1d_t2 import CNN1D_T2
from models.cnn1d_t2_gru import CNN1D_T2_GRU
from models.cnn2d_s3 import CNN2D_S3
from models.cnn3d_s3_t2 import CNN3D_S3_T2
from models.cnn3d_s3_t2_gru import CNN3D_S3_T2_GRU

deep_models_2d = ['mlp', 'cnn1d_t2', 'cnn1d_t2_gru']
deep_models_3d = ['cnn2d_s3', 'cnn3d_s3_t2', 'cnn3d_s3_t2_gru']

def get_model_config_by_name_and_scenario(name, scenario):
    default_lr = 0.001
    models = {
        'mlp': {
            'class_name': MLP,
            'params': {
                'module__n_chans': scenario.get('params', {}).get('n_chans', 48),
                'lr': 0.00001,
                'max_epochs': 500
            }
        },
        'cnn1d_t2': {
            'class_name': CNN1D_T2,
            'params': {
                'module__n_chans': scenario.get('params', {}).get('n_chans', 48),
                'lr': 0.00001,
                 #old 700
                'max_epochs': 1
            }
        },
        'cnn1d_t2_gru': {
            'class_name': CNN1D_T2_GRU,
            'params': {
                'module__n_chans': scenario.get('params', {}).get('n_chans', 48),
                'lr': 0.00001,
                'max_epochs': 600
            }
        },
        'cnn2d_s3': {
            'class_name': CNN2D_S3,
            'params': {
                'module__conv_1_n_features_in': 61,
                'lr': 0.00001,
                'max_epochs': 1000
            }
        },
        'cnn3d_s3_t2': {
            'class_name': CNN3D_S3_T2,
            'params': {
                'lr': 0.00001,
                #old 1200
                'max_epochs': 200
            },
        },
        'cnn3d_s3_t2_gru': {
            'class_name': CNN3D_S3_T2_GRU,
            'params': {
                'lr': 0.00001,
                'max_epochs': 200
            },
        },
        # 'cnn1d_t': {
        #     'class_name': CNN1D_T,
        #     'params': {
        #         'module__n_chans': scenario.get('params', {}).get('n_chans', 56),
        #         'lr': default_lr
        #         # 'module__input_time_length': 61
        #     }
        # },
        # 'cnn1d_gru': {
        #     'class_name': CNN1D_GRU,
        #     'params': {
        #         'module__n_chans': scenario.get('params', {}).get('n_chans', 56),
        #         'lr': default_lr
        #         # 'module__input_time_length': 61
        #     }
        # },
        # 'mlp_tdb': {
        #     'class_name': MLP_TDB,
        #     'params': {
        #         'module__n_chans': scenario.get('params', {}).get('n_chans', 56),
        #         'lr': default_lr
        #         # 'module__input_time_length': 61
        #     }
        # },
        # 'mlp_tdb_lstm': {
        #     'class_name': MLP_TDB_LSTM,
        #     'params': {
        #         'module__n_chans': scenario.get('params', {}).get('n_chans', 56),
        #         'lr': default_lr
        #         # 'module__input_time_length': 61
        #     }
        # },
        # 'cnn2d_s': {
        #     'class_name': CNN2D_S,
        #     'params': {
        #         'module__conv_1_n_features_in': 61,
        #         'lr': default_lr
        #     }
        # },
        # 'cnn2d_s_tdb': {
        #     'class_name': CNN2D_S_TDB,
        #     'params': {
        #         'module__conv_1_n_features_in': 61,
        #         'lr': default_lr
        #     }
        # },
        # 'cnn3d_s_t': {
        #     'class_name': CNN3D_S_T,
        #     'params': {
        #         'module__n_chans': scenario.get('params', {}).get('n_chans', 56),
        #         'lr': default_lr
        #     }
        # }
        # 'eeg_net': {
        #     'class_name': EEG_NET,
        #     'params': {
        #         'module__n_chans': scenario.get('params', {}).get('n_chans', 56),
        #         'lr': default_lr
        #     }
        # }
    }
    return models[name]
