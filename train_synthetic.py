import numpy as np
import argparse
import sys  
import pickle 
import pdb
import time 
import warnings
import os 

# from model.deep_models import deep_classifier
from bootstrap.bootstrap_synthetic import cb_backdoor, cb_frontdoor, cb_front_n_back, cb_par_front_n_back, cb_label_flip
from model.simple_models import simple_models
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
    
parser = argparse.ArgumentParser(description='Causal Weighting techniques')
parser.add_argument('--type','-t', type = str, default='back',required = True)
parser.add_argument('--samples','-N', type = int, default=4000,required = False)
parser.add_argument('--model','-m', type=str, default='log', required=False)
parser.add_argument('--para','-p', type=str, default='log2', required=False)
parser.add_argument('--output_dir', type=str, required = True)
parser.add_argument('--corr-coff','-q', type=float, required=True, default=0.95)
parser.add_argument('--test-corr','-tq',type=float, required=True, default=0.95)
parser.add_argument('--qzy','-z',type=float, required=False, default=0.9)
parser.add_argument('--dim','-d', type=int, default=10, required=False)
args = parser.parse_args()

print(f'{args.corr_coff}, {args.test_corr}')    
warnings.filterwarnings("ignore")

N = args.samples # No of samples
device = 'cuda'

os.makedirs(args.output_dir, exist_ok=True)      

p = 0.5 # p(Y=0)=p(Y=1)=0.5
keylist_train = ["Simple", "IF", "CB", "DA"]
keylist_test = ["Conf", "Unconf", "Reverse_Conf"]

train = dict.fromkeys(keylist_train,None)
test = dict.fromkeys(keylist_test,None)
test_IF = dict.fromkeys(keylist_test,None)

# generating training data for different methods

for trte in ['train', 'test']:

    print(trte)
    if trte == 'train':
        corr = args.corr_coff
    elif trte == 'test': 
        corr = args.test_corr        

    if args.type == 'label_flip':
        simple, CB = cb_label_flip(p,corr,0.8,0.95,args.samples,args.dim)
        unconf, _ = cb_label_flip(p,0.5,0.8,0.95,args.samples,args.dim)
        rev_conf, _ = cb_label_flip(p,-corr,0.8,0.95,args.samples,args.dim)

    if args.type == 'back':
        simple, CB = cb_backdoor(p,corr,args.samples,args.dim) 
        unconf, _ = cb_backdoor(p,0.5,args.samples,args.dim) 
        rev_conf, _ = cb_backdoor(p,-corr,args.samples,args.dim) 
                    
    if args.type == 'front':
        simple, CB = cb_frontdoor(p,corr,args.qzy,args.samples,args.dim)
        unconf, _ = cb_frontdoor(p,0.5,args.qzy,args.samples,args.dim)
        rev_conf, _ = cb_frontdoor(p,-corr,args.qzy,args.samples,args.dim)

    if args.type == 'back_front':
        simple, CB = cb_front_n_back(p,corr,args.qzy,args.samples,args.dim)
        unconf, _ = cb_front_n_back(p,0.5,args.qzy,args.samples,args.dim)
        rev_conf, _ = cb_front_n_back(p,-corr,args.qzy,args.samples,args.dim)

    
    if args.type == 'par_back_front':
        simple, CB = cb_par_front_n_back(p,corr,corr,args.qzy,args.samples,args.dim)
        unconf, _ = cb_par_front_n_back(p,0.5,0.5,args.qzy,args.samples,args.dim)
        rev_conf, _ = cb_par_front_n_back(p,-corr,-corr,args.qzy,args.samples,args.dim)
    
    # import pdb; pdb.set_trace()
    # IF method - adding additional features to input
    if args.type == 'label_flip' or args.type == 'back_front' or args.type == 'par_back_front':
        IF_conf_ip = np.concatenate((simple[0], simple[2], simple[3]), axis=1)
        IF_unconf_ip = np.concatenate((unconf[0], unconf[2], unconf[3]), axis=1)
        IF_revconf_ip = np.concatenate((rev_conf[0], rev_conf[2], rev_conf[3]), axis=1)
    elif args.type == 'back':  
        IF_conf_ip = np.concatenate((simple[0], simple[2]), axis=1)   
        IF_unconf_ip = np.concatenate((unconf[0], unconf[2]), axis=1)
        IF_revconf_ip = np.concatenate((rev_conf[0], rev_conf[2]), axis=1)
    elif args.type == 'front':
        IF_conf_ip = np.concatenate((simple[0], simple[3]), axis=1)
        IF_unconf_ip = np.concatenate((unconf[0], unconf[3]), axis=1)
        IF_revconf_ip = np.concatenate((rev_conf[0], rev_conf[3]), axis=1)

    if trte == "train":
        train["Simple"] = (simple[0],simple[1])
        train["CB"] = (CB[0],CB[1])
        train["DA"] = (unconf[0], unconf[1])
        train['IF'] = (IF_conf_ip, simple[1])
    elif trte == "test": 
        
        test_IF["Conf"] = (IF_conf_ip, simple[1])
        test_IF["Unconf"] = (IF_unconf_ip, unconf[1])
        test_IF["Reverse_Conf"] = (IF_revconf_ip, rev_conf[1])
        
        test["Conf"] = (simple[0], simple[1])
        test["Unconf"] = (unconf[0], unconf[1])
        test["Reverse_Conf"] = (rev_conf[0], rev_conf[1])

# Simple linear classifiers 
Acc = {}
mod = args.model; para = args.para
for tr_type in keylist_train:
    for ts_type in keylist_test:
        if tr_type == "IF":
            Acc[f'{tr_type}:{ts_type}'] = simple_models(mod,para,train[tr_type],test_IF[ts_type]) 
        else:
            Acc[f'{tr_type}:{ts_type}'] = simple_models(mod,para,train[tr_type],test[ts_type])
print(f"Results:{Acc}")

path = os.path.join(args.output_dir,f'fin_res_{args.test_corr}.p')
print(f"saving results pickle at location -> {path}")

with open(path, 'wb') as res:
    pickle.dump(Acc, res)