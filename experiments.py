import numpy as np
from itertools import product
    
def combinations(grid):
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))
        
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].hparams()    


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname
 
    
#### write experiments here    
class Camelyon:
    fname = 'train.py'
    
    @staticmethod
    def hparams():
        grid = {
           'type': ['back', 'front', 'back_front', 'label_flip'],
           'data': ['camelyon'],
           'data_type': ['Conf', 'Deconf', 'DA', 'IF'],
           'domains': ((2, 3),),           
           'corr-coff': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
        }
        
        return combinations(grid)

class CXR:
    fname = 'train.py'
    
    @staticmethod
    def hparams():
        grid = {
           'type': ['back', 'front', 'back_front', 'label_flip'],
           'data': ['CXR'],
           'data_type': ['Conf', 'Deconf', 'DA', 'IF'],    
           'corr-coff': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
           'samples': [6500],
           'use_pretrained': [True],
           'cache_cxr': [True]
        }
        
        return combinations(grid)


class Poverty:
    fname = 'train.py'
    
    @staticmethod
    def hparams():
        grid = {
           'type': ['back', 'front', 'back_front', 'label_flip'],
           'data': ['poverty'],
           'data_type': ['Conf', 'Deconf', 'DA', 'IF'],    
           'corr-coff': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
           'samples': [300],
           'domains': (('malawi', 'kenya', 'tanzania', 'nigeria'),),    
        }
        
        return combinations(grid)

class Synthetic:
    fname = 'train_synthetic.py'
    @staticmethod
    def hparams():
        hps = []
        for corr_coff in list(np.linspace(0.65, 0.95, 4)):
            hps += combinations({
                'type': ['back', 'front', 'back_front', 'par_back_front', 'label_flip'],
                'corr-coff': [corr_coff],
                'test-corr': [corr_coff]
            })
        
        return hps
    
class EnvClf:
    fname = 'train_env.py'
    
    @staticmethod
    def hparams():
        grid = {
           'data': ['CXR', 'camelyon', 'poverty', 'NIH', 'MNIST', 'CelebA'],
           'seed': list(range(5)),
            'use_pretrained': [True] 
        }
        
        return combinations(grid)    
        