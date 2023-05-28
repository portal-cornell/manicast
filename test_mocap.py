# import rospy
import json
# from visualization_msgs.msg import MarkerArray, Marker
# from geometry_msgs.msg import Point
import numpy as np
from model import *
import torch
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.ang2joint import *
from utils.loss_funcs import mpjpe_error, fde_error, weighted_mpjpe_error, perjoint_error
from utils.amass_3d import *
from utils.mocap_3d import Datasets as MoCapDatasets
from utils.transitions_3d import Transitions
from utils.parser import args

mapping_file = "mapping.json"
with open(mapping_file, 'r') as f:
    mapping = json.load(f)       

if __name__ == '__main__':
    model = Model(args.input_dim,args.input_n, args.output_n,args.st_gcnn_dropout,args.joints_to_consider,
                args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to('cpu')
    model_name='amass_3d_'+str(args.output_n)+'frames_ckpt'
    model.load_state_dict(torch.load(f'./checkpoints/{args.load_path}/{args.model_num}_{model_name}'))
    model.eval()

    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
    
    joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints[:-2]])
    
    Dataset_test = MoCapDatasets('./mocap_data',args.input_n,args.output_n,sample_rate=25,split=2)
    loader_test = DataLoader(
        Dataset_test,
        batch_size=args.batch_size,
        shuffle =False,
        num_workers=0)
    
    Dataset_transitions_test = Transitions('./mocap_data',args.input_n,args.output_n,sample_rate=25,split=2)
    loader_transition_test = DataLoader(
        Dataset_transitions_test,
        batch_size=len(Dataset_transitions_test)//len(loader_test),
        shuffle =False,
        num_workers=0)   

    running_loss=0
    running_per_joint_error=0
    running_fde=0
    n=0
    joint_weights = torch.Tensor([1, 1, 1.0, 1, 1,1, 1])
    joint_weights = joint_weights.unsqueeze(0).unsqueeze(0).unsqueeze(3)

    with torch.no_grad():
        for cnt,batch in enumerate(loader_test): 
            batch = batch.float()[:, :, joint_used]
            batch_dim=batch.shape[0]
            n+=batch_dim
            sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
            sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
            # import pdb; pdb.set_trace()
            sequences_predict=model(sequences_train)
            loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
            # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
            per_joint_error=perjoint_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
            fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   
            running_loss += loss*batch_dim
            running_per_joint_error += per_joint_error*batch_dim
            running_fde += fde*batch_dim
    # print('[%d]  testing loss: %.3f' %(epoch + 1, running_loss.item()/n))  
    print('test/mpjpe', running_loss.item()/n)
    print('test/fde', running_fde.item()/n)
    for idx, joint in enumerate(relevant_joints[:-2]):
        print(f'test/{joint}_error', running_per_joint_error[idx].item()/n)

    
    with torch.no_grad():
        for cnt,batch in enumerate(loader_transition_test): 
            batch = batch.float()[:, :, joint_used]
            batch_dim=batch.shape[0]
            n+=batch_dim
            sequences_train=batch[:,0:args.input_n,:,:].permute(0,3,1,2)
            sequences_predict_gt=batch[:,args.input_n:args.input_n+args.output_n,:,:]
            # import pdb; pdb.set_trace()
            sequences_predict=model(sequences_train)
            loss = weighted_mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt, joint_weights)*1000
            # loss=mpjpe_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000 # the inputs to the loss function must have shape[N,T,V,C]
            per_joint_error=perjoint_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
            fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   
            running_loss += loss*batch_dim
            running_per_joint_error += per_joint_error*batch_dim
            running_fde += fde*batch_dim
    # print('[%d]  testing loss: %.3f' %(epoch + 1, running_loss.item()/n))  
    print('transition test/mpjpe', running_loss.item()/n)
    print('transition test/fde', running_fde.item()/n)
    for idx, joint in enumerate(relevant_joints[:-2]):
        print(f'transition test/{joint}_error', running_per_joint_error[idx].item()/n)

# print("Future reaction times = ", np.array(future_reaction_times)- np.array(current_reaction_times))
# print("Current reaction times = ", current_reaction_times)
# print("Forecast reaction times = ", np.array(forecast_reaction_times) - np.array(current_reaction_times))


# import pdb; pdb.set_trace()
