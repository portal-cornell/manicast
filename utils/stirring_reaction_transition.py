import torch
from torch.utils.data import Dataset
from utils.read_json_data import read_json, get_pose_history
import os

stirring_reaction_splits = [
            [
                'stirring_reaction_data/train'
            ],
            [
                'stirring_reaction_data/val'
            ],
            [
                'stirring_reaction_data/test'
            ],
        ]

def convert_time_to_frame(time, hz, offset):
    mins = int(time[:time.find(':')])
    secs = int(time[time.find(':')+1:])
    return ((mins*60 + secs) * hz) + offset

class StirringReactionTransitions(Dataset):
    def __init__(self, data_dir,input_n,output_n,sample_rate, split=0, mocap_splits=stirring_reaction_splits):
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
        self.timestamps = read_json('./mocap_data/stirring_reaction_data/stirring_reaction_transitions.json')
        

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
                for skeleton_name in [self.timestamps[episode]["name"]]:
                    if episode in ignore_data[skeleton_name]:
                        print('Ignoring for ' + skeleton_name)
                        continue
                    tensor = get_pose_history(json_data, skeleton_name)
                    # chop the tensor into a bunch of slices of size sequence_len
                    skip_rate = int(round(120/self.sample_rate))
                    select_frames = torch.tensor(range(len(tensor)//skip_rate))*skip_rate
                    skipped_frames = tensor[select_frames]

                    for (start, end) in self.timestamps[episode]["timestamps"]:
                        start_frame, end_frame = convert_time_to_frame(start, self.sample_rate, 0), convert_time_to_frame(end, self.sample_rate, 0)
                        # print(f'start: {start_frame}, end: {end_frame}')
                        for finish in range(start_frame, end_frame+1):
                            self.data_lst.append(skipped_frames[finish-sequence_len:finish, :, :])
                            # if skipped_frames[begin:begin+sequence_len, :, :].shape[0] != 35:
                            #     print(skipped_frames[begin:begin+sequence_len, :, :].shape)
                            #     print(sequence_len)

        for idx, seq in enumerate(self.data_lst):
            self.data_lst[idx] = seq[:, :, :] - seq[input_n-1:input_n, 21:22, :]
        print(len(self.data_lst))


    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        # each element of the data list is of shape (sequence length, 25 joints, 3d)
        return self.data_lst[idx]
