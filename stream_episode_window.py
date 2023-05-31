import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from model import *
import torch
from utils.parser import args
from pynput import keyboard

def get_relevant_joints(all_joints, relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']):                       
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[mapping[joint]]
        relevant_joint_pos.append(pos)
        # import pdb; pdb.set_trace()
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

def get_forecast(history_joints, future_joints):
    history_joints = torch.Tensor(np.array(history_joints)).unsqueeze(0)
    current_left_hip = history_joints[:,-2:-1,-2:-1,:]
    current_hips = history_joints[:,-2:-1,-2:,:]
    history_joints = history_joints - current_left_hip
    sequences_train=history_joints[:,:,:-2].permute(0,3,1,2)
    with torch.no_grad():
        sequences_predict=model(sequences_train).permute(0,1,3,2)
        if args.prediction_method == "cvm":
            start = sequences_train[:, :, 0:1, :7]
            end = sequences_train[:, :, -1:, :7]
            # import pdb; pdb.set_trace()
            # print(start.shape)
            # input()
            mult = (torch.arange(args.output_n)+1).unsqueeze(0).unsqueeze(0).unsqueeze(3)
            sequences_predict=end+mult*(end-start)/args.input_n
            sequences_predict = sequences_predict.permute(0,2,1,3).permute(0,1,3,2)
        if args.prediction_method == "future":
            sequences_predict = torch.Tensor(future_joints)
            return sequences_predict[0].cpu().numpy()
    current_hips_repeat = current_hips.repeat(1, sequences_predict.shape[1], 1, 1)
    forecast_joints = torch.cat([sequences_predict+current_left_hip, current_hips_repeat], dim=2)
    # import pdb; pdb.set_trace()
    return forecast_joints[0].cpu().numpy()
    # pass

# def get_marker(id, pose, edge, ns = 'current', alpha=1, color=1):
#     marker = Marker()
#     marker.header.frame_id = "mocap"
#     marker.header.stamp = rospy.Time.now()
#     marker.type = marker.LINE_LIST
#     marker.id = id
#     marker.scale.x = 0.005
#     marker.action = marker.ADD 
#     marker.color.b = color
#     marker.color.a = alpha
#     marker.ns = f'{ns}-{relevant_joints[edge[0]]}_{relevant_joints[edge[1]]}'
#     pos1, pos2 = pose[edge[0]], pose[edge[1]]
#     p1, p2 = Point(), Point()
#     x, y, z = pos1.tolist()
#     p1.x, p1.y, p1.z = -x, z, y
#     x, y, z = pos2.tolist()
#     p2.x, p2.y, p2.z = -x, z, y
#     marker.points = [p1, p2]
#     return marker

def get_marker(id, pose, edge, ns = 'current', alpha=1, red=1, green=1, blue=1):
    relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
    SCALE = 0.015
    marker = Marker()
    marker.header.frame_id = "mocap"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.LINE_LIST
    marker.id = id
    marker.scale.x = SCALE
    marker.action = marker.ADD 
    marker.ns = f'{ns}-{relevant_joints[edge[0]]}_{relevant_joints[edge[1]]}'
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = alpha
    p1m = Marker()
    p1m.header.frame_id = "mocap"
    p1m.header.stamp = rospy.Time.now()
    p1m.type = marker.SPHERE_LIST
    p1m.id = id + 101
    p1m.scale.x = SCALE*2
    p1m.scale.y = SCALE*2
    p1m.scale.z = SCALE*2
    p1m.action = p1m.ADD
    p1m.color.r = red
    p1m.color.g = green
    p1m.color.b = blue
    p1m.color.a = alpha/2
    pos1, pos2 = pose[edge[0]], pose[edge[1]]
    p1, p2 = Point(), Point()
    x, y, z = pos1.tolist()
    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker,p1m


def get_marker_array(current_joints, future_joints, forecast_joints, person = "Kushal"):
    id_offset = 100000 if person == 'Kushal' else 0
    color = 1 if person == "Kushal" else 0
    marker_array = MarkerArray()
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6)
        ]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    # for idx, edge in enumerate(edges + extra_edges):
    #     marker_array.markers.append(get_marker(idx+id_offset, 
    #                                            current_joints, 
    #                                            edge,
    #                                            ns=f'current',
    #                                            color=color))
    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(idx, current_joints, edge,ns=f'current', alpha=1, 
                         red=0.1, 
                         green=0.1, 
                         blue=0.0)
        marker_array.markers.append(tup[0])
        marker_array.markers.append(tup[1])

    # for i, time in enumerate([24]):
    #     for idx, edge in enumerate(edges + extra_edges):
    #         marker_array.markers.append(get_marker((i+1)*9+idx, 
    #                                     future_joints[time], 
    #                                     edge, 
    #                                     ns=f'future-{time}', 
    #                                     alpha=0.4-0.1*((time+1)/25),
    #                                     color=0))

    for i, time in enumerate([0,2,4,6,8,10,12,14,16,18,20,22,24]):
        for idx, edge in enumerate(edges + extra_edges):
            tup = get_marker((i+2)*900+idx, 
                                        forecast_joints[time], 
                                        edge,
                                        ns=f'forecast{time}', 
                                        alpha=0.7-0.35*(time+1)/25,
                                        red=0.1, 
                                        green=0.1+0.15*(time+1)/25, 
                                        blue=0.4+0.6*(time+1)/25)
            marker_array.markers.append(tup[0])
            marker_array.markers.append(tup[1])

    # for i, time in enumerate([24]):
    #     for idx, edge in enumerate(edges + extra_edges):
    #         marker_array.markers.append(get_marker((i+2)*900+idx+id_offset, 
    #                                     forecast_joints[time], 
    #                                     edge,
    #                                     , 
    #                                     alpha=0.4-0.1*((time+1)/25),
    #                                     color=1))
    return marker_array



if __name__ == '__main__':
    rospy.init_node('forecaster', anonymous=True)
    human_forecast = rospy.Publisher("/human_forecast", MarkerArray, queue_size=1)

    model = Model(args.input_dim,args.input_n, args.output_n,args.st_gcnn_dropout,args.joints_to_consider,
                args.n_tcnn_layers,args.tcnn_kernel_size,args.tcnn_dropout).to('cpu')
    model_name='amass_3d_'+str(args.output_n)+'frames_ckpt'
    model.load_state_dict(torch.load(f'./checkpoints/{args.load_path}/{args.model_num}_{model_name}'))
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # print(model.eval())
    model.eval()

    
    episode_file = "handover.json"


    episode_file = f"/home/portal/MHAD_Processing/{args.activity}_data/{args.activity}_{args.ep_num}.json"
    mapping_file = "mapping.json"
    with open(episode_file, 'r') as f:
        data = json.load(f)
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
    
    joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints])
    
    pause = False
    def on_press(key):
        if key == keyboard.Key.space:
            pause = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    rate = rospy.Rate(120)

    person_data = {}
    for stream_person in data:
        person_data[stream_person] = np.array(data[stream_person])
    for timestep in range(args.start_frame, args.end_frame):
        print(round((timestep-args.start_frame)/120, 1))
        if not pause and listener.running:
            for stream_person in data:
                if stream_person != "Kushal": continue
                joint_data = person_data[stream_person]
                current_joints = get_relevant_joints(joint_data[timestep])
                history_joints = get_history(joint_data, timestep)
                future_joints = get_future(joint_data, timestep)
                forecast_joints = get_forecast(history_joints, future_joints)
                marker_array = get_marker_array(current_joints=current_joints, 
                                                future_joints=future_joints,
                                                forecast_joints=forecast_joints,
                                                person=stream_person)
                # future_markers = get_future_markers(future_joints)
                human_forecast.publish(marker_array)
                rate.sleep()
        else:
            input("Press enter to continue")
            pause = False
            listener = keyboard.Listener(on_press=on_press)
            listener.start()


# import pdb; pdb.set_trace()
