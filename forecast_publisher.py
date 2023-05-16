#!/usr/bin/env python3
import rospy 
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import String
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from model import *
from utils.loss_funcs import mpjpe_error
import torch
import math

# model_path = '/home/portal/human_forecasting/STSGCN/checkpoints/mocap_new/amass_3d_25frames_ckpt'
model_path = '/home/portal/human_forecasting/Human_Motion_Forecasting/checkpoints/finetune/amass_3d_25frames_ckpt'

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

joint_used = np.array([12, 16, 17, 18, 19, 20, 21])

CYLINDER_POS = [-0.65, 1.05, 1.12]
TABLE_POS = [-.1, 0.9, 1]
cylinder_color = [0,1,0]
radius = 0.45

in_collision = False
FRAME = 'map' #change to mocap when integrating


joint_history = {}
time_step = 0
import time
ct = time.time()
def callback(data):
    global time_step
    global ct
    if time_step not in joint_history:
        joint_history[time_step] = {}
    # print(data.ns)
    if data.ns in joint_history[time_step]: 
        print(data.ns)
        print(time_step)
        print('WHYYYYY')

    joint_history[time_step][data.ns] = data
    if len(joint_history[time_step]) == 25:
        if time_step >= 100:
            del joint_history[time_step - 100]
        time_step += 1
        # if time_step % 120 ==0:
        #     print(time.time()-ct)
        #     print(time_step)

def get_relevant_joints(all_joints):
    prefix = 'Prithwish_'
    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                       'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut',
                       'WaistLBack', 'WaistRBack']
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[prefix + joint].pose.position
        relevant_joint_pos.append([pos.x, pos.y, pos.z])
        # import pdb; pdb.set_trace()
    return relevant_joint_pos
        
def create_input():
    global time_step
    end_time = time_step - 1
    start_time = end_time - 49
    input_array = []
    for i in range(start_time, end_time+1, 5):
        if i < 0:
            idx = 0
        else:
            idx = i
        input_array.append(get_relevant_joints(joint_history[idx]))
    # print(len(input_array))
    return input_array

def publish_forecasts(sequences_predict, current_pose, history_pose):
    global FRAME
    marker_array = MarkerArray()
    # prefix = 'body_'
    # edges = [('RShoulderBack', 'RElbowOut'), ('RElbowOut', 'RWristIn'), 
    #          ('LShoulderBack', 'LElbowOut'), ('LElbowOut', 'LWristIn'),
    #          ('RShoulderBack', 'WaistRFront'), ('RShoulderBack', 'BackTop'),
    #          ('LShoulderBack', 'WaistLFront'), ('LShoulderBack', 'BackTop'),
    #         #  ('BackTop', 'HeadFront'),
    #          ('WaistLFront', 'WaistRFront'),
    #          ('RShoulderBack', 'LShoulderBack')
    #          ]
    # edges = [
    #     ('BackTop', 'LShoulderBack'), ('BackTop', 'RShoulderBack'),
    #     ('LShoulderBack', 'LElbowOut'), ('RShoulderBack', 'RElbowOut'),
    #     ('LElbowOut', 'LWristOut'), ('RElbowOut', 'RWristOut'),
    #     ('WaistLBack', 'WaistRBack'),
    #     ('RShoulderBack', 'WaistRBack'),
    #     ('LShoulderBack', 'WaistLBack'),
    #     ('RShoulderBack', 'LShoulderBack')
    # ]
    # joint_used = np.array([2, 9, 16, 7, 14, 13, 20])
    edges = [
        (0, 1), (0, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 6)
    ]

    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    
    # creates the poses for the true current pose
    # indices 0-8
    for idx, edge in enumerate(edges + extra_edges):
        marker = Marker()
        marker.header.frame_id = FRAME
        marker.type = marker.LINE_LIST
        marker.id = idx
        marker.scale.x = 0.02
        marker.action = marker.ADD 
        # marker.color.b = 0
        marker.color.a = 1
        # print(edge)
        pos1, pos2 = current_pose[edge[0]], current_pose[edge[1]]
        # import pdb; pdb.set_trace()
        p1, p2 = Point(), Point()
        p1.x, p1.y, p1.z = pos1.tolist()
        p2.x, p2.y, p2.z = pos2.tolist()
        # print(p1)
        marker.points = [p1, p2]
        marker_array.markers.append(marker)

    # creates the poses for the history pose
    # indices 9-17
    # for idx, edge in enumerate(edges + extra_edges):
    #     marker = Marker()
    #     marker.header.frame_id = FRAME
    #     marker.type = marker.LINE_LIST
    #     marker.id = 9 + idx
    #     marker.scale.x = 0.02
    #     marker.action = marker.ADD 
    #     marker.color.r = 1
    #     marker.color.g = 1
    #     marker.color.b = 1
    #     marker.color.a = 0.5
    #     # print(edge)
    #     pos1, pos2 = history_pose[edge[0]], history_pose[edge[1]]
    #     # import pdb; pdb.set_trace()
    #     p1, p2 = Point(), Point()
    #     p1.x, p1.z, p1.y = pos1.tolist()
    #     p2.x, p2.z, p2.y = pos2.tolist()
    #     # print(p1)
    #     marker.points = [p1, p2]
    #     marker_array.markers.append(marker)

    

    # connects the forecast base to the current hip positions
    # indices 18-35
    for i, time in enumerate([10, 24]):
        for idx, edge in enumerate(edges):
            marker = Marker()
            marker.header.frame_id = FRAME
            marker.type = marker.LINE_LIST
            marker.id = (i+2)*9 + idx
            marker.scale.x = 0.02
            marker.action = marker.ADD 
            marker.color.b = 1
            marker.color.a = 0.5 + 0.5*(time/25)
            # print(edge)
            pos1, pos2 = current_pose[edge[0]] if edge[0] >= 7 else sequences_predict[time][edge[0]], current_pose[edge[1]] if edge[1] >= 7 else sequences_predict[time][edge[1]]
            # import pdb; pdb.set_trace()
            p1, p2 = Point(), Point()
            p1.x, p1.y, p1.z = pos1.tolist()
            p2.x, p2.y, p2.z = pos2.tolist()
            # print(p1)
            marker.points = [p1, p2]
            marker_array.markers.append(marker)

    determine_cylinder_color(sequences_predict, current_pose)

    
    return marker_array

def determine_cylinder_color(furthest_poses, current_pose):
    global in_collision
    in_collision = False
    cylinder_color[0] = 0
    cylinder_color[1] = 1
    for p in furthest_poses:
        for joint in p:
            x, y, z = joint.tolist()
            if math.sqrt(pow(CYLINDER_POS[0]-x, 2) + pow(CYLINDER_POS[1]-y, 2)) < radius:
                cylinder_color[0] = 1
                cylinder_color[1] = 0
                in_collision = True


    for joint in current_pose:
        x, y, z = joint.tolist()
        if math.sqrt(pow(CYLINDER_POS[0]-x, 2) + pow(CYLINDER_POS[1]-y, 2)) < radius:
            cylinder_color[0] = 1
            cylinder_color[1] = 0
            in_collision = True


    


def get_forecast_array():
    batch = torch.tensor(create_input()).float()
    # print(batch.shape)
    batch = batch.unsqueeze(0)
    # print(batch.shape)
    # current_left_hip = batch[:,-1,-2:-1,:]
    # batch = batch[:, :, :, :] - batch[:, 0:10, -2:-1, :]
    # batch = batch[:, :, :, :] - current_left_hip


    sequences_train=batch[:,:,:-2,:].permute(0,3,1,2)
    sequences_predict=model(sequences_train).permute(0,1,3,2)
    # print(sequences_predict.shape)
    marker_array = publish_forecasts((sequences_predict)[0], (batch[:,-1,:,:])[0], (batch[:,0,:,:])[0])

    # cap_frame_length = len(skipped_frames)

    # input_n = 10
    # output_n = 25

    # joint_used = np.array([12, 16, 17, 18, 19, 20, 21])
    # # joint_used = np.array([2, 9, 16, 7, 14, 13, 20])

    # losses, preds, gts = [], [], []




    # for start in range(input_n, cap_frame_length-input_n-output_n):
    #     batch = torch.from_numpy(skipped_frames[None, start:start+input_n+output_n]).float()
    #     batch = batch[:, :, :, :] - batch[:, 0:1, 21:22, :]
        
    #     sequences_train=batch[:,0:input_n,joint_used,:].permute(0,3,1,2)
    #     sequences_predict_gt=batch[:,:input_n+output_n,:,:]
        
    #     sequences_predict=model(sequences_train).permute(0,1,3,2)
    #     all_joints_seq=sequences_predict_gt.clone()
    #     all_joints_seq[:,input_n:,joint_used,:]=sequences_predict
        
    #     relevant_pred = all_joints_seq[:, input_n:, joint_used, :]
    #     relevant_gt = sequences_predict_gt[:, input_n:, joint_used, :]

    #     loss=mpjpe_error(relevant_pred,relevant_gt)*1000# # both must have format (batch,T,V,C)

    #     data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()
    #     data_gt=torch.squeeze(sequences_predict_gt,0).cpu().data.numpy()
    #     preds.append(data_pred)
    #     gts.append(data_gt)
    #     losses.append(loss.item())



    # marker_array = MarkerArray()
    return marker_array

def get_cylinder():
    global FRAME
    marker = Marker()
    marker.header.frame_id = FRAME
    marker.type = marker.CYLINDER
    marker.ns = 'cylinder'
    marker.id = 100
    marker.scale.x = radius * 2
    marker.scale.y = radius * 2
    marker.scale.z = 0.5
    marker.action = marker.ADD 
    marker.color.r = cylinder_color[0]
    marker.color.g = cylinder_color[1]
    marker.color.b = cylinder_color[2]
    marker.color.a = 0.3
    marker.pose.position.x = CYLINDER_POS[0]
    marker.pose.position.y = CYLINDER_POS[1]
    marker.pose.position.z = CYLINDER_POS[2] + 0.15
    return marker

def get_pot():
    # 113, 121, 126
    global FRAME
    marker = Marker()
    marker.header.frame_id = FRAME
    marker.type = marker.CYLINDER
    marker.ns = 'pot'
    marker.id = 101
    marker.scale.x = radius
    marker.scale.y = radius
    marker.scale.z = radius * 0.5
    marker.action = marker.ADD 
    marker.color.a = 1
    # marker.color.r = 113/255
    # marker.color.g = 121/255
    # marker.color.b = 126/255
    marker.pose.position.x = CYLINDER_POS[0]
    marker.pose.position.y = CYLINDER_POS[1]
    marker.pose.position.z = CYLINDER_POS[2]
    return marker

def get_table():
    global FRAME
    marker = Marker()
    marker.header.frame_id = FRAME
    marker.type = marker.CUBE
    marker.ns = 'table'
    marker.id = 102
    marker.scale.x = radius * 4
    marker.scale.y = radius * 2
    marker.scale.z = radius * 0.15
    marker.action = marker.ADD 
    marker.color.r = 235/255
    marker.color.g = 180/255
    marker.color.b = 131/255
    marker.color.a = 1
    marker.pose.position.x = TABLE_POS[0]
    marker.pose.position.y = TABLE_POS[1]
    marker.pose.position.z = TABLE_POS[2]
    return marker
    


def listener():
    global time_step
    global in_collision
    rospy.init_node('forecaster', anonymous=True)

    rospy.Subscriber("/skeleton_markers", Marker, callback)

    pub = rospy.Publisher("/human_forecast", MarkerArray, queue_size=1)
    # pub2 = rospy.Publisher("/human_forecast_joints", MarkerArray, queue_size=1)
    pub_intersect = rospy.Publisher("/cylinder", Marker, queue_size=1)
    pub_collision = rospy.Publisher("/dora_collision", String, queue_size=1)

    rate = rospy.Rate(120)
    first = True
    pot_marker = get_pot()
    pub_intersect.publish(pot_marker)
    rate.sleep()
    table_marker = get_table()
    pub_intersect.publish(table_marker)
    rate.sleep()
    while not rospy.is_shutdown():
        if time_step == 0:
            continue
        marker_array = get_forecast_array()
        pub.publish(marker_array)
        cylinder_marker = get_cylinder()
        pub_intersect.publish(cylinder_marker)
        pub_collision.publish('COLLISION' if in_collision else 'CONTINUE')
        # pub2.publish(MarkerArray())
        rate.sleep()


    # spin() simply keeps python from exiting until this 
        if first:
            
            first = False
        pub_intersect.publish(pot_marker)
        pub_intersect.publish(table_marker)
        # node is stopped
    # rospy.spin()  


if __name__ == '__main__':
    listener()