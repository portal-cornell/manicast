import json
import numpy as np
from model.manicast import ManiCast
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.ang2joint import *
from utils.loss_funcs import mpjpe_error, fde_error, weighted_mpjpe_error, perjoint_error_eval, perjoint_fde
from data.utils.amass import *
from data.utils.comad import *
from data.utils.transitions import *
from utils.parser import args

mapping_file = "mapping.json"
with open(mapping_file, 'r') as f:
    mapping = json.load(f)       

models = ["pretrained_unweighted"]

model_results_dict = {}
if __name__ == '__main__':
    
    for model_path in models:
        model = ManiCast(args.input_dim,args.input_n, args.output_n,args.st_gcnn_dropout,args.joints_to_consider,
                    args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to('cpu')
        model_name='amass_3d_'+str(args.output_n)+'frames_ckpt'
        if model_path == "current" or model_path == "cvm":
            model.load_state_dict(torch.load(f'./model_checkpoints/SCRATCH/{args.model_num}_{model_name}'))
            args.prediction_method = model_path
        else:
            args.prediction_method = "neural"
            model.load_state_dict(torch.load(f'./model_checkpoints/{model_path}/{args.model_num}_{model_name}'))
        model.eval()

        relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                            'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
        
        joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints[:-2]])
        
        Dataset_test = CoMaD('./data/comad_data',args.input_n,args.output_n,sample_rate=25,split=2)
        loader_test = DataLoader(
            Dataset_test,
            batch_size=args.batch_size,
            shuffle =False,
            num_workers=0)
        
        Dataset_transitions_test = CoMaDTransitions('./data/comad_data',args.input_n,args.output_n,sample_rate=25,split=2)
        loader_transition_test = DataLoader(
            Dataset_transitions_test,
            batch_size=len(Dataset_transitions_test)//len(loader_test),
            shuffle =False,
            num_workers=0)   

        running_loss=0
        running_per_joint_error=0
        running_per_joint_errors = []
        running_per_joint_fdes = []
        running_per_joint_fde=0
        running_fde=0
        n=0
        joint_weights = torch.Tensor([1, 1, 1.0, 1, 1,1, 1])
        joint_weights = joint_weights.unsqueeze(0).unsqueeze(0).unsqueeze(3)

        print("="*20)
        with torch.no_grad():
            for cnt,batch in enumerate(loader_test): 
                batch = batch.float()[:, :, joint_used]
                batch_dim=batch.shape[0]
                n+=batch_dim
                sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
                sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
                if args.prediction_method == "neural":
                    sequences_predict=model(sequences_train)
                elif args.prediction_method == "current":
                    sequences_predict = sequences_train[:, :, -1:, :].repeat(1, 1, args.output_n, 1)
                    sequences_predict = sequences_predict.permute(0,2,1,3)
                elif args.prediction_method == "cvm":
                    start = sequences_train[:, :, 0:1, :]
                    end = sequences_train[:, :, -1:, :]
                    mult = (torch.arange(args.output_n)+1).unsqueeze(0).unsqueeze(0).unsqueeze(3)
                    sequences_predict=end+mult*(end-start)/args.input_n
                    sequences_predict = sequences_predict.permute(0,2,1,3)
                loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
                # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
                per_joint_error, per_joint_error_list=perjoint_error_eval(sequences_predict.permute(0,1,3,2),sequences_predict_gt)
                fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   
                per_joint_fde, per_joint_fde_list=perjoint_fde(sequences_predict.permute(0,1,3,2),sequences_predict_gt)
                running_per_joint_errors+=list(per_joint_error_list.cpu().numpy())
                running_per_joint_fdes+=list(per_joint_fde_list.cpu().numpy())
                running_per_joint_fde += per_joint_fde*batch_dim
                running_loss += loss*batch_dim
                running_per_joint_error += per_joint_error*batch_dim
                running_fde += fde*batch_dim
        all_joints_ade_mean = np.array(running_per_joint_errors).mean(axis=1).mean()*1000
        all_joints_ade_std = np.array(running_per_joint_errors).mean(axis=1).std()*1000/np.sqrt((n))

        all_joints_fde_mean = np.array(running_per_joint_fdes).mean(axis=1).mean()*1000
        all_joints_fde_std = np.array(running_per_joint_fdes).mean(axis=1).std()*1000/np.sqrt((n))

        wrist_ade_mean = np.array(running_per_joint_errors)[:, 5:7].mean(axis=1).mean()*1000
        wrist_ade_std = np.array(running_per_joint_errors)[:, 5:7].mean(axis=1).std()*1000/np.sqrt((n))

        wrist_fde_mean = np.array(running_per_joint_fdes)[:, 5:7].mean(axis=1).mean()*1000
        wrist_fde_std = np.array(running_per_joint_fdes)[:, 5:7].mean(axis=1).std()*1000/np.sqrt((n))

        print("="*20)
        running_loss=0
        running_per_joint_error=0
        running_per_joint_fde=0
        running_per_joint_errors = []
        running_per_joint_fdes = []
        running_fde=0
        n=0
        with torch.no_grad():
            for cnt,batch in enumerate(loader_transition_test): 
                batch = batch.float()[:, :, joint_used]
                batch_dim=batch.shape[0]
                n+=batch_dim
                sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
                sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
                if args.prediction_method == "neural":
                    sequences_predict=model(sequences_train)
                elif args.prediction_method == "current":
                    sequences_predict = sequences_train[:, :, -1:, :].repeat(1, 1, args.output_n, 1)
                    sequences_predict = sequences_predict.permute(0,2,1,3)
                elif args.prediction_method == "cvm":
                    start = sequences_train[:, :, 0:1, :]
                    end = sequences_train[:, :, -1:, :]
                    mult = (torch.arange(args.output_n)+1).unsqueeze(0).unsqueeze(0).unsqueeze(3)
                    sequences_predict=end+mult*(end-start)/args.input_n
                    sequences_predict = sequences_predict.permute(0,2,1,3)
                loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
                # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
                per_joint_error, per_joint_error_list=perjoint_error_eval(sequences_predict.permute(0,1,3,2),sequences_predict_gt)
                fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   
                per_joint_fde, per_joint_fde_list=perjoint_fde(sequences_predict.permute(0,1,3,2),sequences_predict_gt)
                running_per_joint_errors+=list(per_joint_error_list.cpu().numpy())
                running_per_joint_fdes+=list(per_joint_fde_list.cpu().numpy())
                running_per_joint_fde += per_joint_fde*batch_dim
                running_loss += loss*batch_dim
                running_per_joint_error += per_joint_error*batch_dim
                running_fde += fde*batch_dim
        t_wrist_ade_mean = np.array(running_per_joint_errors)[:, 5:7].mean(axis=1).mean()*1000
        t_wrist_ade_std = np.array(running_per_joint_errors)[:, 5:7].mean(axis=1).std()*1000/np.sqrt((n))

        t_wrist_fde_mean = np.array(running_per_joint_fdes)[:, 5:7].mean(axis=1).mean()*1000
        t_wrist_fde_std = np.array(running_per_joint_fdes)[:, 5:7].mean(axis=1).std()*1000/np.sqrt((n))

        model_results_dict[model_path] = {
            'all_joints_ade': [all_joints_ade_mean, all_joints_ade_std],
            'all_joints_fde': [all_joints_fde_mean, all_joints_fde_std],
            'wrist_ade': [wrist_ade_mean, wrist_ade_std],
            'wrist_fde': [wrist_fde_mean, wrist_fde_std],
            't_wrist_ade': [t_wrist_ade_mean, t_wrist_ade_std],
            't_wrist_fde': [t_wrist_fde_mean, t_wrist_fde_std],
        }
    
    metrics = ['all_joints','wrist','t_wrist']
    for metric in metrics:
        for model_path in models:
            ade_metric = metric + '_ade'
            mean = round(model_results_dict[model_path][ade_metric][0], 1)
            std = round(model_results_dict[model_path][ade_metric][1], 1)
            print(f'{mean} (\pm {std})', end = ' &')
            if model_path == "cvm":
                print('&', end='')
        print()
        print('='*20)
        for model_path in models:
            fde_metric = metric + '_fde'
            mean = round(model_results_dict[model_path][fde_metric][0], 1)
            std = round(model_results_dict[model_path][fde_metric][1], 1)
            print(f'{mean} (\pm {std})', end = ' &')
            if model_path == "cvm":
                print('&', end='')
        print()
        print('='*20)


