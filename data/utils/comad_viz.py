import json
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from model.manicast import ManiCast
import torch


def get_relevant_joints(
    all_joints, mapping,
    relevant_joints=[
        "BackTop",
        "LShoulderBack",
        "RShoulderBack",
        "LElbowOut",
        "RElbowOut",
        "LWristOut",
        "RWristOut",
        "WaistLBack",
        "WaistRBack",
    ],
):
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[mapping[joint]]
        relevant_joint_pos.append(pos)
    return relevant_joint_pos


def get_history(all_joints, current_idx, mapping, history_length=10, skip_rate=5):
    history_joints = []
    for i in range(
        current_idx - (history_length - 1) * skip_rate, current_idx + 1, skip_rate
    ):
        idx = max(0, i)
        history_joints.append(get_relevant_joints(all_joints[idx], mapping))
    return history_joints


def get_future(all_joints, current_idx, mapping, future_length=25, skip_rate=5):
    future_joints = []
    for i in range(
        current_idx + skip_rate, current_idx + future_length * skip_rate + 1, skip_rate
    ):
        idx = min(i, all_joints.shape[0] - 1)
        future_joints.append(get_relevant_joints(all_joints[idx], mapping))
    return future_joints


def get_forecast(history_joints, device, model):
    history_joints = torch.Tensor(np.array(history_joints)).unsqueeze(0).to(device)
    current_left_hip = history_joints[:, -2:-1, -2:-1, :].to(device)
    current_hips = history_joints[:, -2:-1, -2:, :].to(device)
    history_joints = history_joints - current_left_hip
    sequences_train = history_joints[:, :, :-2].permute(0, 3, 1, 2)
    with torch.no_grad():
        sequences_predict = model(sequences_train).permute(0, 1, 3, 2)
    current_hips_repeat = current_hips.repeat(1, sequences_predict.shape[1], 1, 1)
    forecast_joints = torch.cat(
        [sequences_predict + current_left_hip, current_hips_repeat], dim=2
    )
    # import pdb; pdb.set_trace()
    return forecast_joints[0].detach().cpu().numpy()


def get_joints(joint_data, timestep, mapping, device, model):
    current_joints = get_relevant_joints(joint_data[timestep], mapping)
    history_joints = get_history(joint_data, timestep, mapping)
    future_joints = get_future(joint_data, timestep, mapping)
    forecast_joints = get_forecast(history_joints, device, model)

    return current_joints, future_joints, forecast_joints


def get_point_array(
    current_joints, future_joints, forecast_joints, figures, ax, timestep, prev, threshold
):
    edges = [(0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6)]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    if current_joints is not None:
        for idx, edge in enumerate(edges + extra_edges):
            pos1, pos2 = current_joints[edge[0]], current_joints[edge[1]]
            x1, y1, z1 = pos1.tolist()
            x2, y2, z2 = pos2.tolist()
            x = np.array([-x1, -x2])
            y = np.array([y1, y2])
            z = np.array([z1, z2])
            if timestep == 0:
                figures[0][0].append(ax.plot(x, z, y, zdir="z", c="black", alpha=1))
                figures[0][1].append(ax.scatter(x, z, y, s=10, c="black", alpha=1))
            else:
                difference_1 = np.array(
                    [
                        (prev[0][idx][0] - x)[0],
                        (prev[0][idx][1] - y)[0],
                        (prev[0][idx][1] - z)[0],
                    ]
                )
                difference_2 = np.array(
                    [
                        (prev[0][idx][0] - x)[1],
                        (prev[0][idx][1] - y)[1],
                        (prev[0][idx][1] - z)[1],
                    ]
                )
                if (
                    np.linalg.norm(difference_1) > threshold
                    or np.linalg.norm(difference_2) > threshold
                ):
                    x, y, z = prev[0][idx]
                figures[0][0][idx][0].set_xdata(x)
                figures[0][0][idx][0].set_ydata(z)
                figures[0][0][idx][0].set_3d_properties(y)

                figures[0][1][idx]._offsets3d = (x, z, y)
            prev[0].append((x, y, z))
    if forecast_joints is not None or future_joints is not None:
        for i, time in enumerate([24]):
            for idx, edge in enumerate(edges + extra_edges):
                if forecast_joints is not None:
                    joints = forecast_joints[time]
                    pos1, pos2 = joints[edge[0]], joints[edge[1]]
                    x1, y1, z1 = pos1.tolist()
                    x2, y2, z2 = pos2.tolist()
                    x = np.array([-x1, -x2])
                    y = np.array([y1, y2])
                    z = np.array([z1, z2])
                    if timestep == 0:
                        figures[1][0].append(
                            ax.plot(
                                x,
                                z,
                                y,
                                zdir="z",
                                c="blue",
                                alpha=0.9 - 0.1 * ((time + 1) / 25),
                            )
                        )
                        figures[1][1].append(
                            ax.scatter(
                                x,
                                z,
                                y,
                                s=10,
                                c="blue",
                                alpha=0.9 - 0.1 * ((time + 1) / 25),
                            )
                        )
                    else:
                        difference_1 = np.array(
                            [
                                (prev[1][idx][0] - x)[0],
                                (prev[1][idx][1] - y)[0],
                                (prev[1][idx][1] - z)[0],
                            ]
                        )
                        difference_2 = np.array(
                            [
                                (prev[1][idx][0] - x)[1],
                                (prev[1][idx][1] - y)[1],
                                (prev[1][idx][1] - z)[1],
                            ]
                        )
                        if (
                            np.linalg.norm(difference_1) > threshold
                            or np.linalg.norm(difference_2) > threshold
                        ):
                            x, y, z = prev[1][idx]

                        figures[1][0][idx][0].set_xdata(x)
                        figures[1][0][idx][0].set_ydata(z)
                        figures[1][0][idx][0].set_3d_properties(y)

                        figures[1][1][idx]._offsets3d = (x, z, y)
                    prev[1].append((x, y, z))
                if future_joints is not None:
                    joints = future_joints[time]
                    pos1, pos2 = joints[edge[0]], joints[edge[1]]
                    x1, y1, z1 = pos1.tolist()
                    x2, y2, z2 = pos2.tolist()
                    x = np.array([-x1, -x2])
                    y = np.array([y1, y2])
                    z = np.array([z1, z2])
                    if timestep == 0:
                        figures[2][0].append(
                            ax.plot(
                                x,
                                z,
                                y,
                                zdir="z",
                                c="green",
                                alpha=0.9 - 0.1 * ((time + 1) / 25),
                            )
                        )
                        figures[2][1].append(
                            ax.scatter(
                                x,
                                z,
                                y,
                                s=10,
                                c="green",
                                alpha=0.9 - 0.1 * ((time + 1) / 25),
                            )
                        )
                    else:
                        difference_1 = np.array(
                            [
                                (prev[2][idx][0] - x)[0],
                                (prev[2][idx][1] - y)[0],
                                (prev[2][idx][1] - z)[0],
                            ]
                        )
                        difference_2 = np.array(
                            [
                                (prev[2][idx][0] - x)[1],
                                (prev[2][idx][1] - y)[1],
                                (prev[2][idx][1] - z)[1],
                            ]
                        )
                        if (
                            np.linalg.norm(difference_1) > threshold
                            or np.linalg.norm(difference_2) > threshold
                        ):
                            x, y, z = prev[2][idx]
                        figures[2][0][idx][0].set_xdata(x)
                        figures[2][0][idx][0].set_ydata(z)
                        figures[2][0][idx][0].set_3d_properties(y)

                        figures[2][1][idx]._offsets3d = (x, z, y)
                    prev[2].append((x, y, z))
