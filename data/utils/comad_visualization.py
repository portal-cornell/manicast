import json
import numpy as np
import argparse
import os
# import torch
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'WaistLBack', 
                        'WaistRBack', 'LHandOut', 'RHandOut']

#How to ask for model paths
# model_path = '/home/portal/Human_Motion_Forecasting/checkpoints/mocap_new/amass_3d_25frames_ckpt'
# model_path = '/home/portal/Human_Motion_Forecasting/checkpoints/finetune_5_1e-03/amass_3d_25frames_ckpt'

# input_dim = 3
# input_n = 10
# output_n = 25
# st_gcnn_dropout = 0.1
# joints_to_consider = 7
# n_tcnn_layers = 4
# tcnn_kernel_size = [3,3]
# tcnn_dropout = 0.0

# model = Model(input_dim, input_n, output_n,st_gcnn_dropout, joints_to_consider,
#               n_tcnn_layers, tcnn_kernel_size, tcnn_dropout).to('cpu')
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# # print(model.eval())
# model.eval()

# GET manicast model

# marginal_model = get_model(ONE_HIST=True, CONDITIONAL=False, bob_hand=False)
# conditional_model = get_model(ONE_HIST=False, CONDITIONAL=True, bob_hand=False)
# model_joints_idx = [0,1,2,3,4,5,6,9,10]

def get_history(joint_data, current_idx, history_length, skip_rate = int(120/15)):
    history_joints = []
    for i in range(current_idx-(history_length-1)*skip_rate, current_idx+1, skip_rate):
        idx = max(0, i)
        history_joints.append(get_relevant_joints(joint_data[idx]))
    return history_joints

def get_relevant_joints(all_joints, relevant_joints = relevant_joints):                       
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[mapping[joint]]
        relevant_joint_pos.append(pos)
    return relevant_joint_pos

def get_history(all_joints, current_idx, history_length=10, skip_rate = 5):
    history_joints = []
    for i in range(current_idx-(history_length-1)*skip_rate, current_idx+1, skip_rate):
        idx = max(0, i)
        history_joints.append(get_relevant_joints(all_joints[idx]))
    return history_joints

def get_future(all_joints, current_idx, future_length=25, skip_rate = 5):
    future_joints = []
    for i in range(current_idx+skip_rate, current_idx + future_length*skip_rate + 1, skip_rate):
        idx = min(i, all_joints.shape[0]-1)
        future_joints.append(get_relevant_joints(all_joints[idx]))
    return future_joints

# def get_forecast(history_joints):
#     history_joints = torch.Tensor(np.array(history_joints)).unsqueeze(0)
#     current_left_hip = history_joints[:,-2:-1,-2:-1,:]
#     current_hips = history_joints[:,-2:-1,-2:,:]
#     history_joints = history_joints - current_left_hip
#     sequences_train=history_joints[:,:,:-2].permute(0,3,1,2)
#     with torch.no_grad():
#         sequences_predict=model(sequences_train).permute(0,1,3,2)
#     current_hips_repeat = current_hips.repeat(1, sequences_predict.shape[1], 1, 1)
#     forecast_joints = torch.cat([sequences_predict+current_left_hip, current_hips_repeat], dim=2)
#     # import pdb; pdb.set_trace()
#     return forecast_joints[0].cpu().numpy()


def get_point_array(current_joints, future_joints, forecast_joints, figure):
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6)
        ]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    if current_joints is not None:
        for idx, edge in enumerate(edges + extra_edges):
            pos1, pos2 = current_joints[edge[0]], current_joints[edge[1]]
            x1,y1,z1= pos1.tolist()
            x2,y2,z2 = pos2.tolist()
            x = [-x1, -x2]
            y = [y1, y2]
            z = [z1, z2]
            figure.plot(x, z, y, zdir='z', c = 'blue', alpha = 1)
    if forecast_joints is not None:
        for i, time in enumerate([24]):
            for idx, edge in enumerate(edges + extra_edges):
                joints = forecast_joints[time]
                pos1, pos2 = joints[edge[0]], joints[edge[1]]
                x1,y1,z1= pos1.tolist()
                x2,y2,z2 = pos2.tolist()
                x = [-x1, -x2]
                y = [y1, y2]
                z = [z1, z2]
                figure.plot(x, z, y, zdir='z', c = 'green', alpha = 0.4-0.1*((time+1)/25))
    if future_joints is not None:
        for i, time in enumerate([24]):
            for idx, edge in enumerate(edges + extra_edges):
                joints = future_joints[time]
                pos1, pos2 = joints[edge[0]], joints[edge[1]]
                x1,y1,z1= pos1.tolist()
                x2,y2,z2 = pos2.tolist()
                x = [-x1, -x2]
                y = [y1, y2]
                z = [z1, z2]
                figure.plot(x, z, y, zdir='z', c = 'yellow', alpha = 0.9-0.1*((time+1)/25))



if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Arguments for running the scripts')
    # parser.add_argument('--dataset',type=str,default="mixing",help="Dataset Type")
    # parser.add_argument('--set_num',type=str,default="0",help="Number of Dataset")
    # parser.add_argument('--ep_num', type=str,default="-1",help="Episode to watch/leave blank if wanting to watch whole set")

    # args = parser.parse_args()


    dataset_folder = f"./data/comad_data/"
    mapping_file = "./mapping.json"

    with open(mapping_file, 'r') as f:
        mapping = json.load(f)


    person_data = {}
    fig = plt.figure(figsize=(10,4.5))
    ax = fig.add_subplot(projection='3d')
    plt.ion()
            

    p_x=np.linspace(-10,10,15)
    p_y=np.linspace(-10,10,15)
    X,Y=np.meshgrid(p_x,p_y)
    # if args.ep_num != "-1":
    # episode_file = f"{dataset_folder}/{args.dataset}_{args.set_num}_{args.ep_num}.json"
    episode_file = "./data/comad_data/chopping_mixing_data/train/chopping_mixing_4.json"
    with open(episode_file, 'r') as f:
        data = json.load(f)
    for stream_person in data:
        person_data[stream_person] = np.array(data[stream_person])
    for timestep in range(len(data[list(data.keys())[0]])):
        print(round(timestep/120, 1))
        joint_data_A = person_data["Kushal"]
        current_joints = get_relevant_joints(joint_data_A[timestep])
        
        history_joints = get_history(joint_data_A, timestep)
        future_joints = get_future(joint_data_A, timestep)
        # forecast_joints = get_forecast(history_joints)
        if (timestep % 120) % 5 == 0:
            plt.cla()
            ax.set_xlim3d([0, 1])
            ax.set_ylim3d([0, 1])
            ax.set_zlim3d([1.2,2.2])
            get_point_array(current_joints=current_joints, future_joints=future_joints, forecast_joints=None, figure=ax)
            plt.title(str(round(timestep/120, 1)),y=-0.1)
        plt.pause(.001)
        if timestep/120 >= 10:
            break
        
    plt.ioff()
    plt.show()
    plt.close()