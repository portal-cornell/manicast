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

model.load_state_dict(torch.load(f"./model_checkpoints/{args.load_path}"))
model.eval()

person_data = {}
fig = plt.figure(figsize=(10, 4.5))
ax = fig.add_subplot(projection="3d")
prev_A = [[], [], []]
prev_B = [[], [], []]
figures_A = [[[], []], [[], []], [[], []]]
figures_B = [[[], []], [[], []], [[], []]]
plt.ion()


p_x = np.linspace(-10, 10, 15)
p_y = np.linspace(-10, 10, 15)
X, Y = np.meshgrid(p_x, p_y)

mapping_file = f"./data/mapping.json"
with open(mapping_file, "r") as f:
    mapping = json.load(f)

episode_file = "./data/comad_data/handover/train/handover_4.json"
with open(episode_file, "r") as f:
    data = json.load(f)

for stream_person in data:
    person_data[stream_person] = np.array(data[stream_person])

ax.set_xlim3d([0, 1])
ax.set_ylim3d([0, 1])
ax.set_zlim3d([1.2, 2.2])
plt.axis("off")
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

    comad_viz.get_point_array(
        current_joints=current_joints_A,
        future_joints=None,
        forecast_joints=forecast_joints_A,
        figures=figures_A,
        timestep=timestep,
        ax=ax,
        prev=prev_A,
        threshold=100,
    )
    comad_viz.get_point_array(
        current_joints=current_joints_B,
        future_joints=None,
        forecast_joints=forecast_joints_B,
        figures=figures_B,
        timestep=timestep,
        ax=ax,
        prev=prev_B,
        threshold=0.8,
    )
    plt.title(str(round(timestep / 120, 1)), y=-0.1)
    plt.title(str(round(timestep / 120, 1)), y=-0.1)
    plt.pause(0.0001)
    # if timestep/120 >= 3:
    #     break

plt.ioff()
plt.show()
plt.close()
