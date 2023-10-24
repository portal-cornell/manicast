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
        if args.prediction_method == "cvm":
            start = sequences_train[:, :, 0:1, :]
            end = sequences_train[:, :, -1:, :]
            mult = (torch.arange(args.output_n)+1).unsqueeze(0).unsqueeze(0).unsqueeze(3)
            sequences_predict=end+mult*(end-start)/args.input_n
            sequences_predict = sequences_predict.permute(0,2,1,3).permute(0,1,3,2)
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

    current_list = []
    forecast_list = []

    threshold = args.threshold
    stream_person = args.stream_person
    joint_data = person_data[stream_person]

    for timestep in range(args.start_frame, args.end_frame):
        current_joints = get_relevant_joints(joint_data[timestep])
        current_pos = np.array(current_joints)[6, :]
        current_list.append(current_pos)

        history_joints = get_history(joint_data, timestep, history_length=args.input_n)
        forecast_joints = get_forecast(history_joints)
        forecast_pos = np.array(forecast_joints)[-1, 6, :]
        forecast_list.append(forecast_pos)


    forecast_time = None
    current_time = None
    for i in range(len(forecast_list)):
        if np.linalg.norm(forecast_list[i]-current_list[-1]) < threshold:
            if forecast_time is None:
                forecast_time = (len(forecast_list)-i)/120
        if np.linalg.norm(current_list[i]-current_list[-1]) < threshold:
            if current_time is None:
                current_time = (len(forecast_list)-i)/120
    
    print("="*20)
    if forecast_time:
        print("DETECTION ADVANTAGE = ", round(forecast_time-current_time,3))
    elif current_time:
        print("GOAL NOT DETECTED")
    print("="*20)

