from torch.utils.data import Dataset,DataLoader
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
from utils.read_json_data import read_json, get_pose_history


class Datasets(Dataset):

    def __init__(self,data_dir,input_n,output_n,sample_rate, split=0):
        """
        data_dir := './mocap_data'
        mapping_file := './mapping.json'
        """
        self.data_dir = data_dir
        self.input_frames = input_n 
        self.output_frames = output_n 
        self.sample_rate = sample_rate
        self.split = split
        self.data_lst = []
        sequence_len = input_n + output_n

        mocap_splits = [
            ['chopping_mixing_data/train'],
            ['chopping_mixing_data/val',],
            ['chopping_mixing_data/test'],
        ]
        names = ["Kushal"]


        for ds in mocap_splits[split]:
            print(f'>>> loading {ds}')
            for episode in os.listdir(self.data_dir + '/' + ds):
                print(f'Episode: {self.data_dir}/{ds}/{episode}')
                json_data = read_json(f'{self.data_dir}/{ds}/{episode}')
                for skeleton_name in names:
                    tensor = get_pose_history(json_data, skeleton_name, self.sample_rate)
                    # chop the tensor into a bunch of slices of size sequence_len
                    self.data_lst.extend(torch.split(tensor, sequence_len)[:-1])
        # if any(t.shape[0] != 35 for t in self.data_lst):
        #     print('last sequence is not long enough')
        for idx, seq in enumerate(self.data_lst):
            self.data_lst[idx] = seq[:, :, :] - seq[input_n-1:input_n, 21:22, :]


    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.data_lst[idx]



if __name__ == "__main__":
    dataset = Datasets('./mocap_data', 10, 25, 25)