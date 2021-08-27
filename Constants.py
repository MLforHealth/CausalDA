from pathlib import Path
import pandas as pd
import numpy as np
import os

wilds_datasets = ['camelyon', 'poverty']

wilds_root_dir = Path('/scratch/hdd001/projects/ml4h/projects/wilds/') ## update
camelyon_path = wilds_root_dir / 'camelyon' ## update
poverty_path = wilds_root_dir / 'poverty' ## update

train_N = {
    'camelyon': 15*4700,
    'CXR': 15*4700,
    'poverty': 1500
}


## CXR
image_paths = {
    'MIMIC': '/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR-JPG', # MIMIC-CXR
    'CXP': '/scratch/hdd001/projects/ml4h/projects/CheXpert/', # CheXpert
    'NIH': '/scratch/hdd001/projects/ml4h/projects/NIH/', # ChestX-ray8
}

df_paths = {
    dataset: {f: os.path.join(image_paths[dataset], 'causalda', f+'.csv') for f in ['train', 'val', 'test']}
    for dataset in image_paths 
}

cache_dir = '/scratch/ssd001/home/haoran/projects/IRM_Clinical/cache' ## update

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema']

