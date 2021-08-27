import Constants
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from pathlib import Path

def preprocess_MIMIC(split, only_frontal):    
    df = split    
    copy_subjectid = df['subject_id'] 
    df.drop(columns = ['subject_id'])
    
    df = df.replace(
            [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
             'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
             '>=90'],
            [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
             'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    
    df['subject_id'] = copy_subjectid.astype(str)
    df = df.rename(
        columns = {
            'Pleural Effusion':'Effusion',   
        })
    df['study_id'] = df['path'].apply(lambda x: x[x.index('p'):x.rindex('/')])
    df['path'] = df['path'].astype(str).apply(lambda x: os.path.join(Constants.image_paths['MIMIC'], x))
    if only_frontal:
        df = df[df.frontal]
        
    df['env'] = 'MIMIC'  
            
    return df[['subject_id','path','env', 'frontal', 'study_id'] + Constants.take_labels]

def preprocess_NIH(split, only_frontal = True):    
    copy_subjectid = split['Patient ID'] 
    split.drop(columns = ['Patient ID'])
    
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
   
    split['subject_id'] = copy_subjectid.astype(str)
    split['path'] = split['Image Index'].astype(str).apply(lambda x: os.path.join(Constants.image_paths['NIH'], 'images', x))
    split['env'] = 'NIH'
    split['frontal'] = True
    split['study_id'] = split['subject_id'].astype(str)
    return split[['subject_id','path', 'env', 'frontal','study_id'] + Constants.take_labels]


def preprocess_CXP(split, only_frontal):    
    copy_subjectid = split['subject_id'] 
    split.drop(columns = ['subject_id'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    
    split['subject_id'] = copy_subjectid.astype(str)
    split = split.rename(
        columns = {
            'Pleural Effusion':'Effusion',
            'Lung Opacity': 'Airspace Opacity'        
        })
    split['path'] = split['Path'].astype(str).apply(lambda x: os.path.join(Constants.image_paths['CXP'], x))
    split['frontal'] = (split['Frontal/Lateral'] == 'Frontal')
    if only_frontal:
        split = split[split['frontal']]
    split['env'] = 'CXP'
    split['study_id'] = split['path'].apply(lambda x: x[x.index('patient'):x.rindex('/')])
    return split[['subject_id','path','env', 'frontal','study_id'] + Constants.take_labels]

def get_process_func(env):
    if env == 'MIMIC':
        return preprocess_MIMIC
    elif env == 'NIH':
        return preprocess_NIH
    elif env == 'CXP':
        return preprocess_CXP
    else:
        raise NotImplementedError        