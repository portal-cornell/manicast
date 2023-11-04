from torch.utils.data import Dataset,DataLoader
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
from utils.read_json_data import read_json, get_pose_history, missing_data

default_splits = [
            [
                'handover/train',
                # 'reactive_stirring/train',
                # 'table_setting/train',
            ],
            [
                'handover/val',
                # 'reactive_stirring/val',
                # 'table_setting/val',
            ],
            [
                'handover/test',
                # 'reactive_stirring/test',
                # 'table_setting/test',
            ],
        ]



default_names = ["Kushal", "Prithwish"]

class CoMaD(Dataset):

    def __init__(self,data_dir,input_n,output_n,sample_rate, split=0, mocap_splits=default_splits, names=default_names):
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
        joint_names = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                      'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut']
        mapping = read_json('./data/mapping.json')
        joint_used = np.array([mapping[joint_name] for joint_name in joint_names])
        

        ignore_data = {
            "Prithwish":['handover_0.json',
                         'handover_2.json',
                         'handover_4.json',
                         'handover_5.json',
                         'handover_8.json',
                         'handover_9.json',
                         'handover_21.json',
                         'handover_23.json'],
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
                    if episode in ignore_data[skeleton_name]:
                        print('Ignoring for ' + skeleton_name)
                        continue
                    tensor = get_pose_history(json_data, skeleton_name)
                    # chop the tensor into a bunch of slices of size sequence_len
                    orig_frames = tensor.shape[0]
                    downsampled_frames = int(round((orig_frames/120)*self.sample_rate))
                    sample_idxs = np.linspace(0, orig_frames-1, downsampled_frames)
                    select_frames = np.round(sample_idxs).astype(int)
                    skipped_frames = tensor[select_frames]
                    for start_frame in range(skipped_frames.shape[0]-sequence_len):
                        end_frame = start_frame + sequence_len
                        if missing_data(skipped_frames[start_frame:end_frame, joint_used, :]):
                            missing_cnt += 1
                            continue
                        self.data_lst.append(skipped_frames[start_frame:end_frame, :, :])
        for idx, seq in enumerate(self.data_lst):
            self.data_lst[idx] = seq[:, :, :] - seq[input_n-1:input_n, 21:22, :]
        print(len(self.data_lst))
        print(f'Missing: {missing_cnt}')


    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.data_lst[idx]



if __name__ == "__main__":
    dataset = Datasets('./data/comad_data', 10, 25, 25)