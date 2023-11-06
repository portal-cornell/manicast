#!/usr/bin/env python3
import rospy 
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from model import *
from utils.loss_funcs import mpjpe_error
import torch
import math


joint_used = np.array([12, 16, 17, 18, 19, 20, 21])

CYLINDER_POS = [-0.75, 1.05, 1.12]
TABLE_POS = [-.1, 0.9, 1]
cylinder_color = [0,1,0]
radius = 0.35

in_collision = False
FRAME = 'mocap' #change to mocap when integrating

joint_history = {}
time_step = 0
import time
ct = time.time()
def callback(data):
    ##3 break down data into a dict with keys, ns (name of person + name of the joint), coordinates, name o
    # data[timestep][name][joint] = coordinate
    global time_step
    global ct
    global joint_history
    
    data_lst = data.data.split(";")
    name = data_lst[2].split("_")[0]
    joint = data_lst[2].split("_")[1]
    coordinates = list(map(float, data_lst[4][2:len(data_lst[4])-1].split(","))) # convert string to list of floats

    if time_step not in joint_history:
        #add time step
        joint_history[time_step] = {}
    if name not in joint_history[time_step]:
        joint_history[time_step][name] = {}
    joint_history[time_step][name][joint] = coordinates
    
    print('----------\n' + str(joint_history))
    #update time ? 
    if len(joint_history[time_step]) == 25:
        time_step += 1
    
    #old:
    # if data.ns in joint_history[time_step]: 
    #     print(data.ns)
    #     print(time_step)
    #     print('WHYYYYY')
    # # print(data.ns)
    # joint_history[time_step][data.ns] = data
    # # if len(joint_history[time_step]) == 25:
    # if len(joint_history[time_step]) == 50:
    #     if time_step >= 100:
    #         del joint_history[time_step - 100]
    #     time_step += 1

def get_relevant_joints(all_joints):
    
    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                       'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut',
                       'WaistLBack', 'WaistRBack', 'LHandOut', 'RHandOut']
    relevant_joint_pos = []

    ##change
    prefix = f'Atiksh_'
    for joint in relevant_joints:
        pos = all_joints[prefix + joint].pose.position
        relevant_joint_pos.append([pos.x, pos.y, pos.z])
    
    prefix = f'Kushal_'
    for joint in relevant_joints:
        pos = all_joints[prefix + joint].pose.position
        relevant_joint_pos.append([pos.x, pos.y, pos.z])

    relevant_joints = ['RPinky', 'RThumb']
    prefix = f'Gripper_'
    for joint in relevant_joints:
        pos = all_joints[prefix + joint].pose.position
        relevant_joint_pos.append([pos.x, pos.y, pos.z])

    return relevant_joint_pos
        
def create_input():
    global time_step
    # print("INPUT TIMESTEP = ", time_step)
    end_time = time_step - 1
    start_time = end_time - 49
    input_array = []
    for i in range(start_time, end_time+1, 5):
        if i < 0:
            idx = 0
        else:
            idx = i
        # idx = end_time
        # print(idx)
        input_array.append(get_relevant_joints(joint_history[idx]))
    # print(len(input_array))
    return input_array

def get_marker(id, pose, edge, ns = 'current', alpha=1, red=1, green=1, blue=1):
    relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 
                        'WaistLBack', 'WaistRBack', 'LHandOut', 'RHandOut']
    SCALE = 0.013
    marker = Marker()
    marker.header.frame_id = FRAME
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
    p1m.header.frame_id = FRAME
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
    # p1.x, p1.y, p1.z = -x+0.2, z-0.6, y+0.1
    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y
    # p2.x, p2.y, p2.z = -x+0.2, z-0.6, y+0.1

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker,p1m


def publish_forecasts(current_joints):
    marker_array = MarkerArray()

    person = "Atiksh"
    id_offset = 100000 if person == 'Atiksh' else 0
    color = 1 if person == "Atiksh" else 0
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6),
            (5, 9), (6, 10)
        ]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(idx+id_offset, current_joints, edge,ns=f'current-Atiksh', alpha=1, 
                         red=0.0, 
                         green=0.0, 
                         blue=0.0)
        marker_array.markers.append(tup[0])
        # marker_array.markers.append(tup[1])
    
    person = "Kushal"
    id_offset = 100000 if person == 'Atiksh' else 0
    color = 1 if person == "Atiksh" else 0
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6),
            (5, 9), (6, 10)
        ]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(idx+id_offset, current_joints[11:], edge,ns=f'current-Kushal', alpha=1, 
                         red=0.0, 
                         green=0.0, 
                         blue=1.0)
        marker_array.markers.append(tup[0])
        # marker_array.markers.append(tup[1])
    return marker_array
    
def get_forecast_array():
    batch = torch.tensor(create_input()).float()
    batch = batch.unsqueeze(0)
    marker_array = publish_forecasts((batch[:,-1,:,:])[0])

    return marker_array


def listener():
    global time_step
    global in_collision
    rospy.init_node('forecaster', anonymous=True)

    rospy.Subscriber("/skeleton_markers", String, callback)

    pub = rospy.Publisher("/human_forecast", MarkerArray, queue_size=1)
    # pub2 = rospy.Publisher("/human_forecast_joints", MarkerArray, queue_size=1)
    # pub_intersect = rospy.Publisher("/cylinder", Marker, queue_size=1)
    # pub_collision = rospy.Publisher("/dora_collision", String, queue_size=1)
    rate = rospy.Rate(120)
    rate.sleep()
    while not rospy.is_shutdown():
        if time_step == 0:
            continue
        marker_array = get_forecast_array()
        pub.publish(marker_array)
        rate.sleep()


if __name__ == '__main__':
    listener()