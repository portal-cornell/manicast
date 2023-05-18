# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
# import ipywidgets as widgets
# from ipywidgets import HBox, VBox, interact
from IPython.display import display
from model import *
from utils.loss_funcs import mpjpe_error, fde_error
import torch
from utils.mocap_3d import Datasets as MoCapDatasets
from utils.amass_3d import Datasets
from torch.utils.data import DataLoader

model_path = './checkpoints/pretrained/amass_3d_25frames_ckpt'

input_dim = 3
input_n = 10
output_n = 25
st_gcnn_dropout = 0.1
joints_to_consider = 7
n_tcnn_layers = 4
tcnn_kernel_size = [3,3]
tcnn_dropout = 0.0

model = Model(input_dim, input_n, output_n,st_gcnn_dropout, joints_to_consider,
              n_tcnn_layers, tcnn_kernel_size, tcnn_dropout).to('cpu')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
print()

Dataset = MoCapDatasets('./mocap_data',10,25,sample_rate=25,split=0)
joint_used = np.array([2, 9, 16, 7, 14, 13, 20])

# Dataset = Datasets('./datasets',10,25,5,split=0)
# joint_used = np.array([12, 16, 17, 18, 19, 20, 21])

loader_val = DataLoader(
        Dataset,
        batch_size=256,
        shuffle = True,
        num_workers=0) 

pretrained = True


lr_lst = [1e-3, 3e-4, 1e-4]
epoch_lst = [5, 10, 15, 20]
import itertools
for lr, epochs in itertools.product(lr_lst, epoch_lst):
    model_path = './checkpoints/finetune_' + str(epochs) + '_' + "%.0e" % lr + '/amass_3d_25frames_ckpt'
    if not pretrained:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    val_loss = []
    val_fde_loss = []
    with torch.no_grad():
        running_loss=0
        running_fde_loss=0
        n=0
        for cnt,batch in enumerate(loader_val): 
            batch = batch.float().to('cpu')[:, :, joint_used]
            batch_dim=batch.shape[0]
            n+=batch_dim
            
            sequences_train=batch[:,0:input_n,:,:].permute(0,3,1,2)
            sequences_predict_gt=batch[:,input_n:input_n+output_n,:,:]
            sequences_predict=model(sequences_train)
            loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
            fde_loss=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
            # if cnt % 200 == 0:
            #     print('[%d, %5d]  validation loss (mpjpe): %.3f' %(epoch + 1, cnt + 1, loss.item()))      
            #     print('[%d, %5d]  validation loss (fde): %.3f' %(epoch + 1, cnt + 1, fde_loss.item()))                
            running_loss+=loss*batch_dim
            running_fde_loss+=fde_loss*batch_dim
        val_loss.append(running_loss.detach().cpu().data.numpy()/n)
        val_fde_loss.append(running_fde_loss.detach().cpu().data.numpy()/n)

    val_loss = np.array(val_loss).mean()
    val_fde_loss = np.array(val_fde_loss).mean()
    print(model_path)
    print(val_loss)
    print(val_fde_loss)
    
    if not pretrained:
        with open('./checkpoints/finetune_' + str(epochs) + '_' + "%.0e" % lr + '/loss.txt', 'w') as f:
            f.write(f'{val_loss},{val_fde_loss}')
            print('wrote')
    else:
        break
