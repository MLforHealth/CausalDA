import numpy as np
import argparse
import sys  
import pickle 
import time 
import os 
import logging 
from timeit import default_timer as timer 
import random 
import pdb 
import os.path as osp 
from pathlib import Path
import math
import json
import Constants

from data.data_cam import Camelyon, split_train_test
from data.data_cam import split_n_label as cam_split_n_label
from data import data_cxr
from utils.logging import setup_logs
from src.training import train_step
from src.validation import validation 
from src.prediction import prediction_analysis
from utils.early_stopping import EarlyStopping 
from utils.checkpointing import save_checkpoint, has_checkpoint, load_checkpoint, delete_checkpoint
from utils.infinite_loader import StatefulSampler, InfiniteDataLoader
from data import data_poverty
from data.data_poverty import split_n_label as poverty_split_n_label

from model import models

## Torch
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision import transforms

# args
parser = argparse.ArgumentParser(description = 'trains a model to predict the environment label')
parser.add_argument('--output_dir','-l', type=str, required=True)
parser.add_argument('--log-interval','-i', type=int, required=False, default=1)
parser.add_argument('--epochs','-e', type=int, required=False, default=15)
parser.add_argument('--val_length', type=int, required=False, default=2048*2)

parser.add_argument('--data','-d', type=str, choices = ['camelyon', 'CXR', 'poverty', 'MNIST', 'NIH', 'CelebA'])
parser.add_argument('--batch-size','-b', type=int, default=64, required=False)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--es_patience', type=int, default=7) # *val_freq steps
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--use_pretrained', action = 'store_true')
parser.add_argument('--cache_cxr', action = 'store_true')
parser.add_argument('--debug', action = 'store_true')

args = parser.parse_args()

run_name = "env" + time.strftime("-%Y-%m-%d_%H_%M_%S")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True    

os.makedirs(args.output_dir, exist_ok=True)
res_pth = os.path.join(args.output_dir, 'results') 
os.makedirs(res_pth, exist_ok=True)

with open(Path(res_pth)/'args.json', 'w') as outfile:
    json.dump(vars(args), outfile)
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu") 
print(device)

batch_size = args.batch_size

class ConcatEnvs(Dataset):
    def __init__(self, ds1, ds2, rotate_second = False):
        self.ds1 = ds1
        self.ds2 = ds2
        self.rotate_second = rotate_second
    
    def __len__(self):
        return len(self.ds1) + len(self.ds2)
    
    def __getitem__(self, idx):
        if idx < len(self.ds1):
            return (self.ds1[idx][0], 0, [])
        else:
            if self.rotate_second:
                return (torch.rot90(self.ds2[idx - len(self.ds1)][0],
                                   1, [1, 2]), 1, [])
            else:
                return (self.ds2[idx - len(self.ds1)][0], 1, [])
        
def random_subset_seq(ds, length):
    return np.random.permutation(len(ds))[:length]

def wilds_get_merged_dataset(WildsDataset, split_n_label, d1, d2, split):
    index_n_labels_1 = split_n_label(split = split, domains = d1).to_numpy()
    index_n_labels_2 = split_n_label(split = split, domains = d2).to_numpy()
    
    return (
        WildsDataset(labels = index_n_labels_1, causal_type = 'none', data_type = 'Conf'),
        WildsDataset(labels = index_n_labels_2, causal_type = 'none', data_type = 'Conf')
    )

def split_dataset(ds, first_frac):
    s1 = random_subset_seq(ds, length = int(first_frac * (len(ds))))
    s2 = np.arange(len(ds))
    s2 = s2[~np.isin(s2, s1)]
    return Subset(ds, s1), Subset(ds, s2)

rotate = True if args.data in ['NIH', 'MNIST', 'CelebA'] else False

if args.data == 'CXR':
    train_e1 = data_cxr.get_dataset(envs = ['MIMIC'], split = 'train', only_frontal = False, cache = args.cache_cxr)
    train_e2 = data_cxr.get_dataset(envs = ['CXP'], split = 'train', only_frontal = False, cache = args.cache_cxr)

    val_e1 = data_cxr.get_dataset(envs = ['MIMIC'], split = 'val', only_frontal = False, cache = args.cache_cxr)    
    val_e2 = data_cxr.get_dataset(envs = ['CXP'], split = 'val', only_frontal = False, cache = args.cache_cxr)   

    test_e1 = data_cxr.get_dataset(envs = ['MIMIC'], split = 'test', only_frontal = False, cache = args.cache_cxr)
    test_e2 = data_cxr.get_dataset(envs = ['CXP'], split = 'test', only_frontal = False, cache = args.cache_cxr)
    
elif args.data in ['poverty', 'camelyon']:    
    if args.data == 'camelyon':
        WildsDataset = Camelyon 
        split_n_label = cam_split_n_label
        d1, d2 = [2], [3]
    else: # poverty
        data_poverty.compute_poverty_split(('malawi', 'kenya', 'tanzania', 'nigeria'))
        WildsDataset = data_poverty.Poverty
        split_n_label = poverty_split_n_label
        d1, d2 = ['malawi', 'tanzania'], ['kenya', 'nigeria']    
    
    train_e1, train_e2 = wilds_get_merged_dataset(WildsDataset, split_n_label, d1, d2, 'train')
    val_e1, val_e2 = wilds_get_merged_dataset(WildsDataset, split_n_label, d1, d2, 'valid')
    test_e1, test_e2 = wilds_get_merged_dataset(WildsDataset, split_n_label, d1, d2, 'test')
    
elif args.data == 'NIH':
    train_e1, train_e2 = split_dataset(data_cxr.get_dataset(envs = ['NIH'], split = 'train', only_frontal = False, cache = args.cache_cxr), 0.5)
    val_e1, val_e2 = split_dataset(data_cxr.get_dataset(envs = ['NIH'], split = 'val', only_frontal = False, cache = args.cache_cxr), 0.5)
    test_e1, test_e2 = split_dataset(data_cxr.get_dataset(envs = ['NIH'], split = 'test', only_frontal = False, cache = args.cache_cxr), 0.5)
    
elif args.data == 'CelebA':
    transform=transforms.Compose([      
        transforms.ToTensor(),
        transforms.Resize(size = [224, 224]),
        transforms.Normalize(Constants.IMAGENET_MEAN, Constants.IMAGENET_STD),
        
        ])
    train_ds = torchvision.datasets.celeba.CelebA('~/datasets', split = 'train', download=False,
                             transform=transform)    
    val_ds = torchvision.datasets.celeba.CelebA('~/datasets', split = 'valid', download=False,
                             transform=transform)    
    test_ds = torchvision.datasets.celeba.CelebA('~/datasets', split = 'test', download=False,
                             transform=transform)
    
    train_e1, train_e2 = split_dataset(train_ds, 0.5)
    val_e1, val_e2 = split_dataset(val_ds, 0.5)
    test_e1, test_e2 =  split_dataset(test_ds, 0.5)
    
elif args.data == 'MNIST':
    transform=transforms.Compose([        
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
    mnist_train_val = torchvision.datasets.MNIST('~/datasets', train=True, download=True,
                             transform=transform)
    train_ds, val_ds = split_dataset(mnist_train_val, 0.8)    
    test_ds = torchvision.datasets.MNIST('~/datasets', train=False, download=True,
                             transform=transform)
    
    train_e1, train_e2 = split_dataset(train_ds, 0.5)
    val_e1, val_e2 = split_dataset(val_ds, 0.5)
    test_e1, test_e2 =  split_dataset(test_ds, 0.5)
    
train_data = ConcatEnvs(train_e1, train_e2, rotate)
valid_data = ConcatEnvs(Subset(val_e1, random_subset_seq(val_e1, args.val_length)),
                       Subset(val_e2, random_subset_seq(val_e2, args.val_length)),
                       rotate)
test_data = ConcatEnvs(test_e1, test_e2, rotate)

train_loader = InfiniteDataLoader(train_data, batch_size=batch_size, num_workers = 1)
validation_loader = DataLoader(valid_data, batch_size=batch_size*2, shuffle=False) 
test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False) 

model_conv = models.get_model(args.data, 'Conf', 'none', args.use_pretrained).to(device)

optimizer = optim.Adam(model_conv.parameters(), lr = args.lr)         
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, factor=0.1)

es = EarlyStopping(patience = args.es_patience)   

if args.debug:
    n_steps = 50
else:
    n_steps = args.epochs * (len(train_data) // batch_size)  
    
if has_checkpoint():
    state = load_checkpoint()
    model_conv.load_state_dict(state['model_dict'])
    optimizer.load_state_dict(state['optimizer_dict'])
    exp_lr_scheduler.load_state_dict(state['scheduler_dict'])
    train_loader.sampler.load_state_dict(state['sampler_dict'])
    start_step = state['start_step']
    es = state['es']
    torch.random.set_rng_state(state['rng'])
    print("Loaded checkpoint at step %s" % start_step)
else:
    start_step = 0    
    
logger = setup_logs(args.output_dir, run_name) # setup logs 
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))
tr_losses, tr_accs = [], []
for step in range(start_step, n_steps):    
    if es.early_stop:
        break     
    data, target, extras = next(iter(train_loader))
    start_timer = timer()

    # Train and validate
    step_loss, step_acc = train_step(data, target, extras, args, model_conv, writer, 
                                 device, optimizer, batch_size) 
    
    end_timer = timer()
    tr_losses.append(step_loss)
    tr_accs.append(step_acc)        

    if step % args.log_interval == 0:
        logger.info('Train Step: {} \tStep time: {:.4f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
            step, end_timer - start_timer, step_acc, step_loss))

    if step % args.val_freq == 0:
        val_acc, val_loss, val_AUC, val_conf_mat, val_F1_score = validation(args, 
            model_conv, device, validation_loader, batch_size)

        writer.add_scalar(f'Loss/train/step', np.mean(tr_losses), step)
        writer.add_scalar(f'Loss/valid/step', val_loss, step)
        writer.add_scalar(f'AUC/valid/step', val_AUC, step)
        writer.add_scalar(f'Accuracy/train/step', np.mean(tr_accs), step)

        tr_losses, tr_accs = [], []

        exp_lr_scheduler.step(val_AUC)

        es(-val_AUC, step , model_conv.state_dict(), Path(res_pth)/'model.pt')   

        save_checkpoint(model_conv, optimizer, exp_lr_scheduler,
                        train_loader.sampler.state_dict(train_loader._infinite_iterator), 
                        step+1, es, torch.random.get_rng_state())
        
model_conv.load_state_dict(torch.load(Path(res_pth)/'model.pt'))

metrics = {}
test_acc, test_loss, AUC, conf_mat, F1_score, target_list, pred_list = prediction_analysis(args, model_conv, device, test_loader, batch_size)   
metrics['target'] = target_list
metrics['pred'] = pred_list
metrics['acc'] = test_acc
metrics['auc'] = AUC
metrics['conf_mat'] = conf_mat

writer.flush()
writer.close()

with open(os.path.join(res_pth, 'final_results.p'), 'wb') as res: 
    pickle.dump(metrics, res)

with open(os.path.join(res_pth, 'done'), 'w') as f:
    f.write('done')    