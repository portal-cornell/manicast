import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from model import *
import torch

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

def get_marker(id, pose, edge, alpha=1, color=1):
    marker = Marker()
    marker.header.frame_id = "mocap"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.LINE_LIST
    marker.id = id
    marker.scale.x = 0.005
    marker.action = marker.ADD 
    marker.color.b = color
    marker.color.a = alpha
    pos1, pos2 = pose[edge[0]], pose[edge[1]]
    p1, p2 = Point(), Point()
    x, y, z = pos1.tolist()
    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y
    marker.points = [p1, p2]
    return marker

def get_marker_array(current_joints, future_joints, forecast_joints):
    marker_array = MarkerArray()
    for idx, edge in enumerate(edges + extra_edges):
        marker_array.markers.append(get_marker(idx, 
                                               current_joints, 
                                               edge))
    for i, time in enumerate([24]):
        for idx, edge in enumerate(edges + extra_edges):
            marker_array.markers.append(get_marker((i+1)*9+idx, 
                                        future_joints[time], 
                                        edge, 
                                        alpha=0.4-0.1*((time+1)/25),
                                        color=0))
    for i, time in enumerate([24]):
        for idx, edge in enumerate(edges + extra_edges):
            marker_array.markers.append(get_marker((i+2)*900+idx, 
                                        forecast_joints[time], 
                                        edge, 
                                        alpha=0.4-0.1*((time+1)/25),
                                        color=1))
    return marker_array

model_folder = '/home/portal/Human_Motion_Forecasting/checkpoints/'
model_path = f'{model_folder}/finetune_5_1e-03/amass_3d_25frames_ckpt'
model_path = f'{model_folder}/finetuned_stirring_reaction_weighted/19_amass_3d_25frames_ckpt'
model_path = f'{model_folder}/finetuned_stirring_reaction_2s/49_amass_3d_25frames_ckpt'
# model_path = /home/portal/Human_Motion_Forecasting/checkpoints/
episode_folder = "/home/portal/Human_Motion_Forecasting/mocap_data/"
activity = "stirring_reaction"
episode_file = f"{episode_folder}/{activity}_data/test/{activity}_4.json"
stream_person = "Kushal"
mapping_file = "/home/portal/Human_Motion_Forecasting/mapping.json"

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

with open(episode_file, 'r') as f:
    data = json.load(f)
with open(mapping_file, 'r') as f:
    mapping = json.load(f)

joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints])
rospy.init_node('forecaster', anonymous=True)
human_forecast = rospy.Publisher("/human_forecast", MarkerArray, queue_size=1)

joint_data = np.array(data[stream_person])
rate = rospy.Rate(1200)

threshold = 0.4

current_reaction_times = []
current_in_danger = False

future_reaction_times = []
future_in_danger = False

forecast_reaction_times = []
forecast_in_danger = False


for timestep in range(joint_data.shape[0]):
    current_joints = get_relevant_joints(joint_data[timestep])
    # import pdb; pdb.set_trace()
    x_max = np.array(current_joints)[:, 0].max()
    if x_max < threshold: current_in_danger=False
    if not current_in_danger and x_max > threshold:
        current_reaction_times.append(timestep/120.0)
        current_in_danger=True
    
    history_joints = get_history(joint_data, timestep)
    future_joints = get_future(joint_data, timestep)

    x_max = np.array(future_joints)[-1, :, 0].max()
    if x_max < threshold: future_in_danger=False
    if not future_in_danger and x_max > threshold:
        future_reaction_times.append(timestep/120.0)
        future_in_danger=True

    forecast_joints = get_forecast(history_joints)

    x_max = np.array(forecast_joints)[:, :, 0].max()
    if x_max < threshold: forecast_in_danger=False
    if not forecast_in_danger and x_max > threshold:
        forecast_reaction_times.append(timestep/120.0)
        forecast_in_danger=True

    marker_array = get_marker_array(current_joints=current_joints, 
                                    future_joints=future_joints,
                                    forecast_joints=forecast_joints)
    # future_markers = get_future_markers(future_joints)
    human_forecast.publish(marker_array)
    # rate.sleep()
print("Future reaction times = ", future_reaction_times)
print("Current reaction times = ", current_reaction_times)
print("Forecast reaction times = ", forecast_reaction_times)


# import pdb; pdb.set_trace()
