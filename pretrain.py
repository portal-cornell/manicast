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
from torch.utils.tensorboard import SummaryWriter
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device: %s'%device)


def train(model, writer, joint_used, joint_names, model_name, joint_weights):
    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    
    Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=0)
    loader_train = DataLoader(
        Dataset,
        batch_size=args.batch_size,
        shuffle = True,
        num_workers=0)    

    Dataset_val = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=1)
    loader_val = DataLoader(
        Dataset_val,
        batch_size=args.batch_size,
        shuffle = True,
        num_workers=0)      

    Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=2)
    loader_test = DataLoader(
        Dataset,
        batch_size=args.batch_size,
        shuffle =False,
        num_workers=0)                    

    model.train()
    for epoch in range(args.n_epochs):
        running_loss=0
        running_per_joint_error=0
        running_fde=0
        n=0
        model.train()
        for cnt,batch in enumerate(loader_train): 
            batch = batch.float().to(device)[:, :, joint_used] # multiply by 1000 for milimeters
            batch_dim=batch.shape[0]
            n+=batch_dim
            sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
            sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
            optimizer.zero_grad()
            sequences_predict=model(sequences_train)
            # import pdb; pdb.set_trace()
            loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
            # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
            per_joint_error=perjoint_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
            fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000          
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
            optimizer.step()
            running_loss += loss*batch_dim
            running_per_joint_error += per_joint_error*batch_dim
            running_fde += fde*batch_dim
        if args.use_scheduler:
            scheduler.step()
        print('[%d]  training loss: %.3f' %(epoch + 1, running_loss.item()/n))  
        writer.add_scalar('train/mpjpe', running_loss.item()/n, epoch)
        writer.add_scalar('train/fde', running_fde.item()/n, epoch)
        for idx, joint in enumerate(joint_names):
            writer.add_scalar(f'train/{joint}_error', running_per_joint_error[idx].item()/n, epoch)
        
        model.eval()
        running_loss=0
        running_per_joint_error=0
        running_fde=0
        n=0
        with torch.no_grad():
            for cnt,batch in enumerate(loader_val): 
                batch = batch.float().to(device)[:, :, joint_used]
                batch_dim=batch.shape[0]
                n+=batch_dim
                sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
                sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
                sequences_predict=model(sequences_train)
                loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
                # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
                per_joint_error=perjoint_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
                fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   
                running_loss += loss*batch_dim
                running_per_joint_error += per_joint_error*batch_dim
                running_fde += fde*batch_dim
        print('[%d]  validation loss: %.3f' %(epoch + 1, running_loss.item()/n))  
        writer.add_scalar('val/mpjpe', running_loss.item()/n, epoch)
        writer.add_scalar('val/fde', running_fde.item()/n, epoch)
        for idx, joint in enumerate(joint_names):
            writer.add_scalar(f'val/{joint}_error', running_per_joint_error[idx].item()/n, epoch)
        
        running_loss=0
        running_per_joint_error=0
        running_fde=0
        n=0
        with torch.no_grad():
            for cnt,batch in enumerate(loader_test): 
                batch = batch.float().to(device)[:, :, joint_used]
                batch_dim=batch.shape[0]
                n+=batch_dim
                sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
                sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
                sequences_predict=model(sequences_train)
                loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
                # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
                per_joint_error=perjoint_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
                fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   
                running_loss += loss*batch_dim
                running_per_joint_error += per_joint_error*batch_dim
                running_fde += fde*batch_dim
        print('[%d]  testing loss: %.3f' %(epoch + 1, running_loss.item()/n))  
        writer.add_scalar('test/mpjpe', running_loss.item()/n, epoch)
        writer.add_scalar('test/fde', running_fde.item()/n, epoch)
        for idx, joint in enumerate(joint_names):
            writer.add_scalar(f'test/{joint}_error', running_per_joint_error[idx].item()/n, epoch)

        print('----saving model-----')
        
        pathlib.Path('./checkpoints/'+args.model_path).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join('./checkpoints/'+args.model_path,f'{epoch}_{model_name}'),
                   )

if __name__ == '__main__':
    # python pretrain.py --input_n 25 --model_path pretrained_unweighted_hist25 --weight 1
    weight = args.weight
    joint_weights_base = torch.tensor([1, 1, 1, 1, 1, weight, weight]).float().to(device)
    joint_weights = joint_weights_base.unsqueeze(0).unsqueeze(0).unsqueeze(3)
    joint_used = np.array([12, 16, 17, 18, 19, 20, 21])
    joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                      'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut']
    model = Model(args.input_dim,args.input_n,
                           args.output_n,args.st_gcnn_dropout,args.joints_to_consider,args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to(device)
    model_name='amass_3d_'+str(args.output_n)+'frames_ckpt'
    print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    writer = SummaryWriter(log_dir='./pretrain_logs/' + args.model_path)
    train(model, writer, joint_used, joint_names, model_name, joint_weights)