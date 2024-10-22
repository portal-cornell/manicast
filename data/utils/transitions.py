import torch
from torch.utils.data import Dataset
from utils.read_json_data import read_json, get_pose_history, missing_data
import os
import numpy as np

splits = [
    [
        "handover/train",
        "reactive_stirring/train",
        "table_setting/train",
    ],
    [
        "handover/val",
        "reactive_stirring/val",
        "table_setting/val",
    ],
    [
        "handover/test",
        "reactive_stirring/test",
        "table_setting/test",
    ],
]


def convert_time_to_frame(time, hz, offset):
    mins = int(time[: time.find(":")])
    secs = int(time[time.find(":") + 1 :])
    return ((mins * 60 + secs) * hz) + offset


class CoMaDTransitions(Dataset):
    def __init__(
        self, data_dir, input_n, output_n, sample_rate, split=0, mocap_splits=splits
    ):
        """
        data_dir := './data/comad_data'
        mapping_file := './data/mapping.json'
        """
        self.data_dir = data_dir
        self.input_frames = input_n
        self.output_frames = output_n
        self.sample_rate = sample_rate
        self.split = split
        self.data_lst = []
        sequence_len = input_n + output_n
        self.timestamps = {
            f"{self.data_dir}/reactive_stirring": read_json(
                f"{self.data_dir}/reactive_stirring/reactive_stirring_transitions.json"
            ),
            f"{self.data_dir}/handover": read_json(
                f"{self.data_dir}/handover/handover_transitions.json"
            ),
        }
        joint_names = [
            "BackTop",
            "LShoulderBack",
            "RShoulderBack",
            "LElbowOut",
            "RElbowOut",
            "LWristOut",
            "RWristOut",
        ]
        mapping = read_json("./data/mapping.json")
        joint_used = np.array([mapping[joint_name] for joint_name in joint_names])

        ignore_data = {
            "Prithwish": [
                "handover_0.json",
                "handover_2.json",
                "handover_4.json",
                "handover_5.json",
                "handover_8.json",
                "handover_9.json",
            ],
            "Kushal": [],
        }

        missing_cnt = 0
        for ds in mocap_splits[split]:
            print(f">>> loading {ds}")
            for episode in os.listdir(self.data_dir + "/" + ds):
                print(f"Episode: {self.data_dir}/{ds}/{episode}")
                json_data = read_json(f"{self.data_dir}/{ds}/{episode}")
                activity_folder = f"{self.data_dir}/{ds}"
                activity_folder = activity_folder[: activity_folder.rfind("/")]
                if (
                    activity_folder in self.timestamps
                ):  # this branch is for the activities where there are a few discrete transitions
                    ts = self.timestamps[activity_folder]
                    for skeleton_name in (
                        [ts[episode]["name"]]
                        if "reaction" in episode
                        else ["Prithwish", "Kushal"]
                    ):
                        if episode in ignore_data[skeleton_name]:
                            print("Ignoring for " + skeleton_name)
                            continue
                        tensor = get_pose_history(json_data, skeleton_name)

                        # print(tensor.shape)
                        orig_frames = tensor.shape[0]
                        downsampled_frames = int(
                            round((orig_frames / 120) * self.sample_rate)
                        )
                        sample_idxs = np.linspace(
                            0, orig_frames - 1, downsampled_frames
                        )
                        select_frames = np.round(sample_idxs).astype(int)
                        skipped_frames = tensor[select_frames]
                        # print(skipped_frames.shape)

                        for start, end in ts[episode]["timestamps"]:
                            start_frame, end_frame = convert_time_to_frame(
                                start, self.sample_rate, 0
                            ), convert_time_to_frame(end, self.sample_rate, 0)
                            for begin in range(
                                max(start_frame - input_n, 0),
                                end_frame + 1 - sequence_len,
                            ):
                                # if skipped_frames[begin:begin+sequence_len, :, :].shape[0] != 35:
                                #     print(f'Begin: {begin}\nLast: {end_frame+1-sequence_len}\n{skipped_frames.shape[0]}')
                                #     print(skipped_frames[begin:begin+sequence_len, :, :].shape)
                                #     print(episode)
                                if missing_data(
                                    skipped_frames[
                                        begin : begin + sequence_len, joint_used, :
                                    ]
                                ):
                                    missing_cnt += 1
                                    continue

                                self.data_lst.append(
                                    skipped_frames[begin : begin + sequence_len, :, :]
                                )

                else:  # this branch is for table setting, where the entirety of the episode has transitions for both people
                    for skeleton_name in ["Prithwish", "Kushal"]:
                        if episode in ignore_data[skeleton_name]:
                            print("Ignoring for " + skeleton_name)
                            continue
                        tensor = get_pose_history(json_data, skeleton_name)
                        skip_rate = int(round(120 / self.sample_rate))
                        select_frames = (
                            torch.tensor(range(len(tensor) // skip_rate)) * skip_rate
                        )
                        skipped_frames = tensor[select_frames]
                        for start_frame in range(
                            skipped_frames.shape[0] - sequence_len
                        ):
                            end_frame = start_frame + sequence_len
                            if missing_data(
                                skipped_frames[start_frame:end_frame, joint_used, :]
                            ):
                                missing_cnt += 1
                                continue
                            self.data_lst.append(
                                skipped_frames[start_frame:end_frame, :, :]
                            )

        for idx, seq in enumerate(self.data_lst):
            self.data_lst[idx] = seq[:, :, :] - seq[input_n - 1 : input_n, 21:22, :]
        print(len(self.data_lst))
        print(f"Missing {missing_cnt}")

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.data_lst[idx]
