from models.riemann import RiemannTangLog
from models.riemann import RiemannXdawnTangLog
from models.riemann import RiemannMDM
from models.riemann import RiemannXdawnMDM

from models.riemann import LDA
from models.riemann import RegLDA
from models.riemann import XdawnRegLDA

riemann_models = ['riemann_tang_log', 'riemann_xdawn_tang_log', 'riemann_mdm', 'riemann_xdawn_mdm', 'lda', 'reg_lda', 'xdawn_reg_lda']

def get_model_config_by_name_and_scenario(name, scenario):
    models = {
        'riemann_tang_log': {
            'class_name': RiemannTangLog,
            'params': {}
        },
        'riemann_xdawn_tang_log': {
            'class_name': RiemannXdawnTangLog,
            'params': {
                'n_components': 2
            }
        },
        'riemann_mdm': {
            'class_name': RiemannMDM,
            'params': {
                'n_components': 2
            }
        },
        'riemann_xdawn_mdm': {
            'class_name': RiemannXdawnMDM,
            'params': {
                'n_components': 2
            }
        },
        'lda': {
            'class_name': LDA,
            'params': {}
        },
        'reg_lda': {
            'class_name': RegLDA,
            'params': {}
        },
        'xdawn_reg_lda': {
            'class_name': XdawnRegLDA,
            'params': {}
        }
    }
    return models[name]
