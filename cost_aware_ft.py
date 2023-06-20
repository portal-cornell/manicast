import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import *
from utils.ang2joint import *
from utils.loss_funcs import mpjpe_error, fde_error, weighted_mpjpe_error, perjoint_error
from utils.amass_3d import *
from utils.parser import args
from utils.mocap_3d import Datasets as MoCapDatasets
from utils.costs_3d import Datasets as CostDatasets
from utils.transitions_3d import Transitions
from utils.read_json_data import read_json
from torch.utils.tensorboard import SummaryWriter
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device: %s'%device)

if __name__ == '__main__':
    Dataset = CostDatasets('./mocap_data',args.input_n,args.output_n,sample_rate=25,split=0)