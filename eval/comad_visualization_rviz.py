import rospy
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import json
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from model.manicast import ManiCast
import torch
from utils.parser import args
from data.utils import comad_viz

device = "cuda"

relevant_joints = [
    "BackTop",
    "LShoulderBack",
    "RShoulderBack",
    "LElbowOut",
    "RElbowOut",
    "LWristOut",
    "RWristOut",
    "WaistLBack",
    "WaistRBack",
]
model = ManiCast(
    args.input_dim,
    args.input_n,
    args.output_n,
    args.st_gcnn_dropout,
    args.joints_to_consider,
    args.n_tcnn_layers,
    args.tcnn_kernel_size,
    args.tcnn_dropout,
).to(device)

model_name = "amass_3d_" + str(args.output_n) + "frames_ckpt"


def get_marker(
    id,
    pose,
    edge,
    ns="current",
    alpha=1,
    red=1,
    green=1,
    blue=1,
    relevant_joints=relevant_joints,
):
    SCALE = 0.015
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.LINE_LIST
    marker.id = id
    marker.scale.x = SCALE
    marker.action = marker.ADD
    marker.ns = f"{ns}-{relevant_joints[edge[0]]}_{relevant_joints[edge[1]]}"
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = alpha
    p1m = Marker()
    p1m.header.frame_id = "map"
    p1m.header.stamp = rospy.Time.now()
    p1m.type = marker.SPHERE_LIST
    p1m.id = id + 101
    p1m.scale.x = SCALE * 2
    p1m.scale.y = SCALE * 2
    p1m.scale.z = SCALE * 2
    p1m.action = p1m.ADD
    p1m.color.r = red
    p1m.color.g = green
    p1m.color.b = blue
    p1m.color.a = alpha / 2
    pos1, pos2 = pose[edge[0]], pose[edge[1]]
    p1, p2 = Point(), Point()
    x, y, z = pos1.tolist()

    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker, p1m


def get_marker_array(current_joints, future_joints, forecast_joints, person="Kushal"):
    id_offset = 100000 if person == "Kushal" else 0
    color = 1 if person == "Kushal" else 0
    marker_array = MarkerArray()
    edges = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6)]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]

    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(
            idx + id_offset,
            current_joints,
            edge,
            ns=f"current",
            alpha=1,
            red=0.0,
            green=0.0,
            blue=0.0,
        )
        marker_array.markers.append(tup[0])
        marker_array.markers.append(tup[1])

    if forecast_joints is not None or future_joints is not None:
        for i, time in enumerate([24]):
            for idx, edge in enumerate(edges):
                if forecast_joints is not None:
                    tup = get_marker(
                        idx + 100 * timestep + id_offset,
                        forecast_joints[timestep],
                        edge,
                        ns=f"forecast",
                        alpha=0.9 - 0.1 * ((time + 1) / 25),
                        red=0.1,
                        green=0.1,
                        blue=0.8,
                    )
                    marker_array.markers.append(tup[0])
                    marker_array.markers.append(tup[1])
                if future_joints is not None:
                    tup = get_marker(
                        idx + 1000000 * timestep + id_offset,
                        future_joints[timestep],
                        edge,
                        ns=f"future",
                        alpha=0.9 - 0.1 * ((time + 1) / 25),
                        red=0.1,
                        green=0.8,
                        blue=0.1,
                    )
                    marker_array.markers.append(tup[0])
                    marker_array.markers.append(tup[1])

    return marker_array

if name == "__main__":
    model.load_state_dict(torch.load(f"./model_checkpoints/{args.load_path}"))
    model.eval()

    rospy.init_node("forecaster", anonymous=True)
    human_A_forecast = rospy.Publisher("/alice_forecast", MarkerArray, queue_size=1)
    human_B_forecast = rospy.Publisher("/bob_forecast", MarkerArray, queue_size=1)

    rate = rospy.Rate(120)

    dataset_folder = f"./data/comad_data/"
    mapping_file = "./mapping.json"

    with open(mapping_file, "r") as f:
        mapping = json.load(f)

    person_data = {}

    episode_file = "./data/comad_data/handover/train/handover_4.json"
    with open(episode_file, "r") as f:
        data = json.load(f)
    for stream_person in data:
        person_data[stream_person] = np.array(data[stream_person])

    for timestep in range(0, len(data[list(data.keys())[0]]), 5):
        print(round(timestep / 120, 1))
        joint_data_A = person_data["Kushal"]
        joint_data_B = person_data["Prithwish"]
        current_joints_A, future_joints_A, forecast_joints_A = comad_viz.get_joints(
            joint_data_A, timestep, mapping, device, model
        )
        current_joints_B, future_joints_B, forecast_joints_B = comad_viz.get_joints(
            joint_data_B, timestep, mapping, device, model
        )

        marker_array_A = get_marker_array(
            current_joints=current_joints_A,
            future_joints=None,
            forecast_joints=forecast_joints_A,
            person="Kushal",
        )
        marker_array_B = get_marker_array(
            current_joints=current_joints_B,
            future_joints=None,
            forecast_joints=forecast_joints_B,
            person="Prithwish",
        )

        human_A_forecast.publish(marker_array_A)
        human_B_forecast.publish(marker_array_B)

        rate.sleep()
