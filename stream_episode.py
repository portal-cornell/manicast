import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from model import *
import torch
from pynput import keyboard

model_path = '/home/portal/datavisualization/Human_Motion_Forecasting/checkpoints/mocap_new/amass_3d_25frames_ckpt'
model_path = '/home/portal/datavisualization/Human_Motion_Forecasting/checkpoints/finetune_5_1e-03/amass_3d_25frames_ckpt'
model_path = '/home/portal/datavisualization/Human_Motion_Forecasting/checkpoints/finetuned_stirring_unweighted_with_transitions/19_amass_3d_25frames_ckpt'

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
# print(model.eval())
model.eval()

episode_file = "/home/portal/datavisualization/Human_Motion_Forecasting/mocap_data/stirring_reaction_data/val/stirring_reaction_15.json"
stream_person = "Prithwish"
mapping_file = "/home/portal/datavisualization/Human_Motion_Forecasting/mapping.json"
with open(episode_file, 'r') as f:
    data = json.load(f)
with open(mapping_file, 'r') as f:
    mapping = json.load(f)

relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                       'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
edges = [
        (0, 1), (0, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 6)
    ]
# extra edges to connect the pose back to the hips
extra_edges = [(1, 7), (7, 8), (8, 2)]


def get_relevant_joints(all_joints):                       
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
        print(idx)
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
    # import pdb; pdb.set_trace()
    return forecast_joints[0].cpu().numpy()
    # pass

def get_marker(id, pose, edge, alpha=1, red=1, green=1, blue=1):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.LINE_LIST
    marker.id = id
    marker.scale.x = 0.01
    marker.action = marker.ADD 
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = alpha
    p1m = Marker()
    p1m.header.frame_id = "map"
    p1m.header.stamp = rospy.Time.now()
    p1m.type = marker.SPHERE_LIST
    p1m.id = id + 101
    p1m.scale.x = .025
    p1m.scale.y = .025
    p1m.scale.z = .025
    p1m.action = p1m.ADD
    p1m.color.r = red
    p1m.color.g = green
    p1m.color.b = blue
    p1m.color.a = alpha
    pos1, pos2 = pose[edge[0]], pose[edge[1]]
    p1, p2 = Point(), Point()
    x, y, z = pos1.tolist()
    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker,p1m

# Colored Gradients (currently blue focused)
def get_marker_array(current_joints, future_joints, forecast_joints):
    marker_array = MarkerArray()
    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(idx, current_joints, edge,alpha=1, red=0.1, green=0.1, blue=.7**((13)**.01+1))
        marker_array.markers.append(tup[0])
        marker_array.markers.append(tup[1])
    for i, time in enumerate([0, 2,4,6,8,10,12,14,16,18,20,22,24]):
        for idx, edge in enumerate(edges + extra_edges):
            tup = get_marker((i+2)*900+idx, 
                                        forecast_joints[time], 
                                        edge, 
                                        alpha=1-0.4*((time+1)/25),
                                        red=0.1, green=0.1+0.1*((time+1)/25),blue=(.7+0.001*((time+1)/25))**((12-i)**.01+1))
            marker_array.markers.append(tup[0])
            marker_array.markers.append(tup[1])

    return marker_array

# B&W Color Scheme
# def get_marker_array(current_joints, future_joints, forecast_joints):
#     marker_array = MarkerArray()
#     for idx, edge in enumerate(edges + extra_edges):
#         tup = get_marker(idx, current_joints, edge, red=1, green=1, blue=1)
#         marker_array.markers.append(tup[0])
#         marker_array.markers.append(tup[1])
#     for i, time in enumerate([2,4,6,8,10,12,14,16,18,20,22,24]):
#         for idx, edge in enumerate(edges + extra_edges):
#             tup = get_marker((i+2)*900+idx, 
#                                         forecast_joints[time], 
#                                         edge, 
#                                         alpha=0.5-0.1*((time+1)/25),
#                                         red=(1-0.2*i*((time+1)/25))**(i**.1+1), green=(1-0.2*i*((time+1)/25))**(i**.1+1),blue=(1-0.2*i*((time+1)/25))**(i**.1+1))
#             marker_array.markers.append(tup[0])
#             marker_array.markers.append(tup[1])

#     return marker_array

joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints])
rospy.init_node('forecaster', anonymous=True)
human_forecast = rospy.Publisher("/human_forecast", MarkerArray, queue_size=1)

joint_data = np.array(data[stream_person])
rate = rospy.Rate(120)
pause = False
def on_press(key):
    if key == keyboard.Key.space:
        pause = True
        return False

listener = keyboard.Listener(on_press=on_press)
listener.start()
for timestep in range(joint_data.shape[0]):
    if not pause and listener.running:
        current_joints = get_relevant_joints(joint_data[timestep])
        history_joints = get_history(joint_data, timestep)
        future_joints = get_future(joint_data, timestep)
        forecast_joints = get_forecast(history_joints)
        marker_array = get_marker_array(current_joints=current_joints, 
                                        future_joints=future_joints,
                                        forecast_joints=forecast_joints)
        # future_markers = get_future_markers(future_joints)
        human_forecast.publish(marker_array)
        rate.sleep()
    else:
        input("Press enter to continue")
        pause = False
        listener = keyboard.Listener(on_press=on_press)
        listener.start()


# import pdb; pdb.set_trace()
