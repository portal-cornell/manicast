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
from utils.loss_funcs import mpjpe_error, fde_error, weighted_mpjpe_error, perjoint_error, perjoint_fde
from utils.amass_3d import *
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

    joint_used = np.array([12, 16, 17, 18, 19, 20, 21])
    joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                      'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut']
    
    Dataset = Datasets(args.data_dir,args.input_n,args.output_n,args.skip_rate,split=2)
    loader_test = DataLoader(
        Dataset,
        batch_size=args.batch_size,
        shuffle =False,
        num_workers=0)

    running_loss=0
    running_per_joint_error=0
    running_fde=0
    running_per_joint_fde=0
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
            per_joint_error=perjoint_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
            per_joint_fde=perjoint_fde(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000
            fde=fde_error(sequences_predict.permute(0,1,3,2),sequences_predict_gt)*1000   

            running_per_joint_fde += per_joint_fde*batch_dim
            running_loss += loss*batch_dim
            running_per_joint_error += per_joint_error*batch_dim
            running_fde += fde*batch_dim
    # print('[%d]  testing loss: %.3f' %(epoch + 1, running_loss.item()/n))  
    print('test/mpjpe', running_loss.item()/n)
    print('test/fde', running_fde.item()/n)
    for idx, joint in enumerate(joint_names):
        if 'Wrist' in joint:
            print(f'test/{joint}_error', running_per_joint_error[idx].item()/n)
    for idx, joint in enumerate(joint_names):
        if 'Wrist' in joint:
            print(f'test/{joint}_fde', running_per_joint_fde[idx].item()/n)

# print("Future reaction times = ", np.array(future_reaction_times)- np.array(current_reaction_times))
# print("Current reaction times = ", current_reaction_times)
# print("Forecast reaction times = ", np.array(forecast_reaction_times) - np.array(current_reaction_times))


# import pdb; pdb.set_trace()
