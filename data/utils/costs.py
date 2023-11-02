from torch.utils.data import Dataset,DataLoader
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
from utils.read_json_data import read_json, get_pose_history, missing_data

default_splits = [
            [
                # 'handover/train',
                'reactive_stirring/train',
                # 'table_setting/train',
            ],
            [
                # 'handover/val',
                'reactive_stirring/val',
                # 'table_setting/val',
            ],
            [
                # 'handover/test',
                'reactive_stirring/test',
                # 'table_setting/test',
            ],
        ]



default_names = ["Kushal", "Prithwish"]

class CostDataset(Dataset):

    def __init__(self,data_dir,input_n,output_n,sample_rate, split=0, mocap_splits=default_splits, names=default_names):
        """
        data_dir := './data/comad_data'
        mapping_file := './mapping.json'
        """
        self.data_dir = data_dir
        self.input_frames = input_n 
        self.output_frames = output_n 
        self.sample_rate = sample_rate
        self.split = split
        self.data_lst = []
        self.other_lst = []
        self.is_reaching_lst = []
        sequence_len = input_n + output_n
        joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                      'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack']
        mapping = read_json('./mapping.json')
        joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        stirring_react_metadata = read_json(f'{self.data_dir}/reactive_stirring/reactive_stirring_metadata.json')

        ignore_data = {
            "Prithwish":['handover_0.json',
                         'handover_2.json',
                         'handover_4.json',
                         'handover_5.json',
                         'handover_8.json',
                         'reactive_stirring_0.json',
                         'reactive_stirring_12.json',
                         'reactive_stirring_14.json'],
            "Kushal":['table_setting_4.json',
                      'table_setting_5.json',
                      'table_setting_6.json',
                      'table_setting_7.json',
                      'table_setting_8.json',
                      'table_setting_9.json',
                      'table_setting_10.json',]
        }

        missing_cnt = 0
        for ds in mocap_splits[split]:
            print(f'>>> loading {ds}')
            for episode in os.listdir(self.data_dir + '/' + ds):
                print(f'Episode: {self.data_dir}/{ds}/{episode}')
                json_data = read_json(f'{self.data_dir}/{ds}/{episode}')
                for skeleton_name in names:
                # for skeleton_name in ["Prithwish", "Kushal"]:
                    is_reaching = stirring_react_metadata[episode]['reaching_human'] == skeleton_name
                    other_name = [name for name in names if name != skeleton_name][0]
                    if episode in ignore_data[skeleton_name] or episode in ignore_data:
                        print('Ignoring for ' + skeleton_name)
                        continue
                    # import pdb; pdb.set_trace()
                    tensor_skeleton = get_pose_history(json_data, skeleton_name)
                    tensor_other = get_pose_history(json_data, other_name)
                    skeleton_frames = self.get_downsampled_frames(tensor_skeleton)
                    other_frames = self.get_downsampled_frames(tensor_other)
                    # chop the tensor into a bunch of slices of size sequence_len
                    for start_frame in range(skeleton_frames.shape[0]-sequence_len):
                        end_frame = start_frame + sequence_len
                        if missing_data(skeleton_frames[start_frame:end_frame, joint_used, :]) or\
                            missing_data(other_frames[start_frame+input_n:end_frame, joint_used, :]):
                            missing_cnt += 1
                            # print("MISSED you")
                            continue
                        self.data_lst.append(skeleton_frames[start_frame:end_frame, :, :])
                        self.other_lst.append(other_frames[start_frame+input_n:end_frame, :, :])
                        self.is_reaching_lst.append(is_reaching)
        # for idx, seq in enumerate(self.data_lst):
        #     self.data_lst[idx] = seq[:, :, :] - seq[input_n-1:input_n, 21:22, :]
        print(len(self.data_lst))
        print(f'Missing: {missing_cnt}')

    def get_downsampled_frames(self, tensor):
        orig_frames = tensor.shape[0]
        downsampled_frames = int(round((orig_frames/120)*self.sample_rate))
        sample_idxs = np.linspace(0, orig_frames-1, downsampled_frames)
        select_frames = np.round(sample_idxs).astype(int)
        skipped_frames = tensor[select_frames]
        return skipped_frames
    
    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.data_lst[idx], self.other_lst[idx], self.is_reaching_lst[idx]



if __name__ == "__main__":
    dataset = Datasets('./data/comad_data', 10, 25, 25)