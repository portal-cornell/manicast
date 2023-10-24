import json
import numpy as np
from model.manicast import ManiCast
import torch
from utils.parser import args
mapping_file = "mapping.json"
with open(mapping_file, 'r') as f:
    mapping = json.load(f)       

def get_relevant_joints(all_joints):
    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                       'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']         
     
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

def get_forecast(history_joints):
    history_joints = torch.Tensor(np.array(history_joints)).unsqueeze(0)
    current_left_hip = history_joints[:,-2:-1,-2:-1,:]
    current_hips = history_joints[:,-2:-1,-2:,:]
    history_joints = history_joints - current_left_hip
    sequences_train=history_joints[:,:,:-2].permute(0,3,1,2)
    with torch.no_grad():
        sequences_predict=model(sequences_train).permute(0,1,3,2)
    current_hips_repeat = current_hips.repeat(1, sequences_predict.shape[1], 1, 1)
    forecast_joints = torch.cat([sequences_predict+current_left_hip, current_hips_repeat], dim=2)
    return forecast_joints[0].cpu().numpy()

if __name__ == '__main__':
    model = ManiCast(args.input_dim,args.input_n, args.output_n,args.st_gcnn_dropout,args.joints_to_consider,
                args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to('cpu')
    model_name='amass_3d_'+str(args.output_n)+'frames_ckpt'
    model.load_state_dict(torch.load(f'./checkpoints/{args.load_path}/{args.model_num}_{model_name}'))
    model.eval()
    episode_file = f"./comad_data/{args.activity}_data/test/{args.activity}_{args.ep_num}.json"
    with open(episode_file, 'r') as f:
        data = json.load(f)
    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
    
    joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints])
    
    pause = False
    person_data = {}
    for stream_person in data:
        person_data[stream_person] = np.array(data[stream_person])

    current_stop_times = []
    current_start_times = []
    current_in_danger = False

    forecast_stop_times = []
    forecast_start_times = []
    forecast_in_danger = False

    threshold = args.threshold
    stream_person = args.stream_person
    joint_data = person_data[stream_person]

    for timestep in range(0, joint_data.shape[0], 5):
        current_joints = get_relevant_joints(joint_data[timestep])
        x_max = np.array(current_joints)[:, 0].max()
        if x_max < threshold and current_in_danger: 
            current_start_times.append(timestep/120.0)
            current_in_danger=False
        if not current_in_danger and x_max > threshold:
            current_stop_times.append(timestep/120.0)
            current_in_danger=True
        
        history_joints = get_history(joint_data, timestep, history_length=args.input_n)
        forecast_joints = get_forecast(history_joints)
        x_max = np.array(forecast_joints)[-1, 6, 0].max()
        if x_max < threshold and forecast_in_danger: 
            forecast_start_times.append(timestep/120.0)
            forecast_in_danger=False
        if not forecast_in_danger and x_max > threshold:
            forecast_stop_times.append(timestep/120.0)
            forecast_in_danger=True

    def rounder(my_list):
        return [ round(elem, 2) for elem in my_list ]
    print(rounder(current_start_times))
    print(rounder(forecast_start_times))

    print(rounder(current_stop_times))
    print(rounder(forecast_stop_times))

    restart_times = []
    for cs in current_start_times:
        for fs in forecast_start_times:
            if fs < cs and fs > cs -1:
                restart_times.append(cs-fs)
                break
    
    stop_times = []
    for cs in current_stop_times:
        for fs in forecast_stop_times:
            if fs < cs and fs > cs -1:
                stop_times.append(cs-fs)
                break
    print(np.mean(np.array(restart_times)))
    print(np.mean(np.array(stop_times)))

