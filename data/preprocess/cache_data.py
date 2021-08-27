#!/h/haoran/anaconda3/bin/python
#SBATCH --partition cpu
#SBATCH --qos nopreemption
#SBATCH -c 4
#SBATCH --mem=20gb

import os
import sys
sys.path.append(os.getcwd())
from pathlib import Path
from Constants import *
import numpy as np
from data.preprocess import validate
from data import cxr_process as process
from data import data_cxr
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type = int, required = False)
args = parser.parse_args()

validate.validate_splits()
i=0
dss = []
for env in df_paths:    
    if args.env_id is None or args.env_id == i:
        ds = data_cxr.get_dataset(envs = [env], split = 'all', augment = -1, cache = True, only_frontal = False)
        print(env)
        dss.append(ds)        
    i += 1

for ds in dss:        
    for i in range(len(ds)):
        ds[i]