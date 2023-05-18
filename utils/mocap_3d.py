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
            ['chopping_mixing_data/train',
             'chopping_mixing_data/val',
             'chopping_mixing_data/test',
             'chopping_stirring_data/train',
             'chopping_stirring_data/val',
             'chopping_stirring_data/test',],
            ['chopping_mixing_data/val',
             'chopping_stirring_data/val',],
            ['chopping_mixing_data/test'],
        ]
        names = ["Kushal", "Prithwish"]

        ignore_data = {
            "Prithwish":['chopping_mixing_0.json',
                         'chopping_mixing_2.json',
                         'chopping_mixing_4.json',
                         'chopping_mixing_5.json',
                         'chopping_mixing_8.json',
                         'chopping_stirring_0.json'],
            "Kushal":[]
        }


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
                    for start_frame in range(tensor.shape[0]-(sequence_len*self.sample_rate)):
                        end_frame = start_frame + (sequence_len*self.sample_rate)
                        self.data_lst.append(tensor[start_frame:end_frame, :, :])
        for idx, seq in enumerate(self.data_lst):
            self.data_lst[idx] = seq[:, :, :] - seq[input_n-1:input_n, 21:22, :]


    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        skip_rate = int(round(120/self.sample_rate))
        select_frames = torch.tensor(range(len(self.data_lst[idx])//skip_rate))*skip_rate
        skipped_frames = self.data_lst[idx][select_frames]
        return skipped_frames



if __name__ == "__main__":
    dataset = Datasets('./mocap_data', 10, 25, 25)