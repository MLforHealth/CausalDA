import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
from tqdm import tqdm
import pdb
import numpy as np
import pandas as pd 
from torch.autograd import Variable
import pickle
import pdb
from model import resnet_multispectral

def get_model(dataset, causal_type, data_type, use_pretrained, n_outputs = 2):
    if data_type == 'IF':
        if causal_type == 'back':
            n_extras = 1
        else:
            n_extras = 2
    else:
        n_extras = 0
        
    if dataset == 'poverty':
        return resnet_multispectral.ResNet18(n_extras)
    elif dataset == 'MNIST':
        return BasicCNN(n_extras)
    else:
        return Conv_conf_emb(use_pretrained, n_extras, n_outputs)
    

class Conv_conf_emb(nn.Module):
    def __init__(self, use_pretrained, num_extra, n_outputs = 2):
        super(Conv_conf_emb, self).__init__()
        self.num_extra = num_extra
        self.model_conv = mod.densenet121(pretrained= use_pretrained)         
        self.num_ftrs = self.model_conv.classifier.in_features + num_extra
        self.model_conv.classifier = nn.Identity()
        self.class_conf =  nn.Linear(self.num_ftrs,n_outputs)

    def forward(self,x,*args):
        assert(len(args) <= 1) 
        img_conv_out = self.model_conv(x)
        if self.num_extra:
            assert(args[0].shape[1] == self.num_extra)        
            img_conv_out = torch.cat((img_conv_out, args[0]), -1)
        out = self.class_conf(img_conv_out)
        return out

class BasicCNN(nn.Module):
    def __init__(self, num_extra):
        super().__init__()
        self.num_extra = num_extra
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.clf = nn.Sequential(
            nn.Linear(in_features=12*4*4, out_features=120 + self.num_extra),
            nn.ReLU(),
            nn.Linear(in_features=120 + self.num_extra, out_features=60),
            nn.ReLU(),
            nn.Linear(in_features=60, out_features=2)
        )        
        
        
    def forward(self, t,*args):
        assert(len(args) <= 1) 
        t = self.encoder(t).reshape(-1, 12*4*4)
        
        if self.num_extra:
            assert(args[0].shape[1] == self.num_extra)        
            t = torch.cat((t, args[0]), -1)

        return self.clf(t) 

