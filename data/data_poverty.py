import PIL
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import Dataset
from torchvision.transforms import transforms  
from PIL import Image 

import os
import os.path as osp 
import torch
import pandas as pd
import numpy as np
import random
from scipy.misc import imread 
from sklearn.model_selection import StratifiedShuffleSplit

import pdb
import pickle

from tqdm import tqdm 
import multiprocessing as mp
from multiprocessing import Pool
import Constants
from pathlib import Path

splits = {} # country: {'train': arr, 'valid': arr, 'test': arr}
tr_domains = set()
test_domains = set()

# compute median poverty index on training set to discretize label
def compute_poverty_split(train_domains):
    tr_domains.clear()
    test_domains.clear()    
    tr_domains.update(train_domains)
    conf_1 = train_domains[::2]
    dhs_df = (pd.read_csv(Constants.poverty_path/'dhs_metadata.csv')
              .reset_index()
              .rename(columns = {'index': 'fidx'})
             )
    poverty_thres_cutoff = dhs_df.loc[dhs_df.country.isin(train_domains), 'wealthpooled'].median()
    dhs_df['target'] = (dhs_df['wealthpooled'] >= poverty_thres_cutoff).astype(int)
    dhs_df['conf'] = (dhs_df['country'].isin(conf_1)).astype(int)
    
    for country in dhs_df.country.unique():
        splits[country] = {}
        temp = dhs_df[dhs_df.country == country]
        if country in train_domains:
            train_idx, val_test_idx  = list(StratifiedShuffleSplit(n_splits=1, train_size= 0.7)
                                            .split(np.zeros(len(temp)), temp['target']))[0] # different split for each seed
            train_df = temp.iloc[train_idx]
            val_test_df = temp.iloc[val_test_idx]
            
            val_idx, test_idx = list(StratifiedShuffleSplit(n_splits=1, test_size=0.5)
                                     .split(np.zeros(len(val_test_idx)), val_test_df['target']))[0]
            val_df = val_test_df.iloc[val_idx]
            test_df = val_test_df.iloc[test_idx]
            
            splits[country]['train'] = df_to_labels(train_df, poverty_thres_cutoff)  
            splits[country]['valid'] = df_to_labels(val_df, poverty_thres_cutoff)  
            splits[country]['test'] = df_to_labels(test_df, poverty_thres_cutoff)  
        else:
            test_domains.add(country)
            splits[country]['test'] = df_to_labels(temp, poverty_thres_cutoff)        
    
def df_to_labels(df, cutoff):
    return df[['fidx', 'target', 'conf']].rename(columns = {'fidx': 'filename',
                               'target': 'label'})
    
def split_n_label(split, domains):
    # returns: array of (index, label, conf)
    if set(domains).issubset(tr_domains): # ID
        return pd.concat([splits[country][split] for country in domains], axis = 0, ignore_index=True)
    else: # hack so that any other argument to train_domains will return OOD
        return pd.concat([splits[country]['test'] for country in test_domains], axis = 0, ignore_index=True)
    
    
class Poverty(Dataset):    
    def __init__(self, labels, causal_type, data_type):        
        self.labels = labels
        self.img_pth = Constants.poverty_path/'landsat_poverty_imgs.npy'
        self.imgs = np.load(self.img_pth, mmap_mode='r').transpose((0, 3, 1, 2))
        self.causal_type = causal_type
        self.data_type = data_type
        self.cache_counter = 0
        
    def __len__(self):
        return len(self.labels)
    
    def load_img(self, img_idx):     
        img = self.imgs[img_idx].copy()
        self.cache_counter += 1
        if self.cache_counter > 1000:
            self.imgs = np.load(self.img_pth, mmap_mode='r').transpose((0, 3, 1, 2))
            self.cache_counter = 0
            
        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        img_idx = self.labels[idx][0]
        img = self.load_img(img_idx)                
        label = torch.tensor(int(self.labels[idx][1]))
        
        if self.data_type == 'IF':
            if self.causal_type == 'back':
                extras = np.array([float(self.labels[idx][2])])
            else:
                extras = np.array([float(self.labels[idx][2]), float(self.labels[idx][3])])
        else:
            extras = np.array([])
        
        return img, label, extras    