# import rospy
import json
# from visualization_msgs.msg import MarkerArray, Marker
# from geometry_msgs.msg import Point
import numpy as np
from model import *
import torch
import matplotlib.pyplot as plt
from utils.loss_funcs import mpjpe_error, perjoint_error

relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                       'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
edges = [
        (0, 1), (0, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 6)
    ]
# extra edges to connect the pose back to the hips
extra_edges = [(1, 7), (7, 8), (8, 2)]


def get_relevant_joints(all_joints):                       
    relevant_joint_pos = []
    for joint in relevant_joints:
        pos = all_joints[mapping[joint]]
        relevant_joint_pos.append(pos)
        # import pdb; pdb.set_trace()
    return relevant_joint_pos

def get_history(all_joints, current_idx, history_length=10, skip_rate = 5):
    history_joints = []
    for i in range(current_idx-(history_length-1)*skip_rate, current_idx+1, skip_rate):
        idx = max(0, i)
        history_joints.append(get_relevant_joints(all_joints[idx]))
    return history_joints

def get_future(all_joints, current_idx, future_length=25, skip_rate = 5):
    future_joints = []
    for i in range(current_idx+skip_rate, current_idx + future_length*skip_rate + 1, skip_rate):
        idx = min(i, all_joints.shape[0]-1)
        future_joints.append(get_relevant_joints(all_joints[idx]))
    return future_joints

def get_forecast(history_joints, model):
    history_joints = torch.Tensor(np.array(history_joints)).unsqueeze(0)
    current_left_hip = history_joints[:,-2:-1,-2:-1,:]
    current_hips = history_joints[:,-2:-1,-2:,:]
    history_joints = history_joints - current_left_hip
    sequences_train=history_joints[:,:,:-2].permute(0,3,1,2)
    with torch.no_grad():
        sequences_predict=model(sequences_train).permute(0,1,3,2)
    current_hips_repeat = current_hips.repeat(1, sequences_predict.shape[1], 1, 1)
    forecast_joints = torch.cat([sequences_predict+current_left_hip, current_hips_repeat], dim=2)
    # import pdb; pdb.set_trace()
    return forecast_joints[0].cpu().numpy()
    # pass

episode_num = 17
model_folder = './checkpoints'
episode_folder = "./mocap_data"
activity = "stirring_reaction"
episode_file = f"{episode_folder}/{activity}_data/test/{activity}_{episode_num}.json"
stream_person = "Kushal"
mapping_file = "./mapping.json"

input_dim = 3
input_n = 10
output_n = 25
st_gcnn_dropout = 0.1
joints_to_consider = 7
n_tcnn_layers = 4
tcnn_kernel_size = [3,3]
tcnn_dropout = 0.0

def create_model(model_path):
    model = Model(input_dim,input_n, output_n,st_gcnn_dropout, joints_to_consider,
                n_tcnn_layers, tcnn_kernel_size, tcnn_dropout).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model

model_map = {
    'AMASS': create_model(f'{model_folder}/pretrained_unweighted/49_amass_3d_25frames_ckpt'),
    # TODO: Create model trained on only our data
    # 'FT-No-Trans': create_model(f'{model_folder}/all_finetuned_unweighted_hist10_no_transitions_1e-04/49_amass_3d_25frames_ckpt'),
    # 'FT-Mix-Trans': create_model(f'{model_folder}/all_finetuned_unweighted_hist10_mixed_transitions_1e-04/49_amass_3d_25frames_ckpt'),
    'FT-Mix-Trans-Wrist6': create_model(f'{model_folder}/all_finetuned_wrist6_hist10_mixed_transitions_1e-04/49_amass_3d_25frames_ckpt')
}
color_map = {
    'Current': '#fc8d62',
    'Future': '#e41a1c',
    'AMASS': '#377eb8',
    # TODO: Create model trained on only our data
    # 'FT-Mix-Trans': '#4daf4a',
    'FT-Mix-Trans-Wrist6': '#4daf4a'
}




with open(episode_file, 'r') as f:
    data = json.load(f)
with open(mapping_file, 'r') as f:
    mapping = json.load(f)

joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints])

joint_data = np.array(data[stream_person])[120*7:]

threshold = 0.5

current_reaction_times = []
current_in_danger = False

future_reaction_times = []
future_in_danger = False

forecast_reaction_times = {model_name: [] for model_name in model_map.keys()}
forecast_in_danger = {model_name: False for model_name in model_map.keys()}

current_x_values = []
future_x_values = []
forecast_x_values = {model_name: [] for model_name in model_map.keys()}
forecast_losses = {model_name: [] for model_name in model_map.keys()}
time_values = []

print(joint_data.shape)
step_interval = 2

for timestep in range(joint_data.shape[0]):
    if (timestep%step_interval) == 0:
        time_values.append(timestep / 120.0)

    current_joints = get_relevant_joints(joint_data[timestep])
    x_max = np.array(current_joints)[[6], 0].max()
    if (timestep%step_interval) == 0:
        current_x_values.append(x_max)
    if x_max < threshold: current_in_danger=False
    if not current_in_danger and x_max > threshold:
        current_reaction_times.append(timestep/120.0)
        current_in_danger=True
    
    history_joints = get_history(joint_data, timestep)
    future_joints = get_future(joint_data, timestep)

    x_max = np.array(future_joints)[-1, [6], 0].max()
    if (timestep%step_interval) == 0:
        future_x_values.append(x_max)
    if x_max < threshold: future_in_danger=False
    if not future_in_danger and x_max > threshold:
        future_reaction_times.append(timestep/120.0)
        future_in_danger=True

    for model_name, model in model_map.items():
        history_joints = get_history(joint_data, timestep, history_length=25 if ('25' in model_name) else 10)
        forecast_joints = get_forecast(history_joints, model)
        x_max = np.array(forecast_joints)[-1, [6], 0].max()
        if (timestep%step_interval) == 0:
            seq_pred = torch.tensor(forecast_joints[:-2]).unsqueeze(0)
            seq_gt = torch.tensor(future_joints[:-2]).unsqueeze(0)
            loss = mpjpe_error(seq_pred,seq_gt)*1000
            per_joint_error=perjoint_error(seq_pred,seq_gt)*1000
            forecast_losses[model_name].append(loss) # MPJPE
            # forecast_losses[model_name].append(per_joint_error[6])
            forecast_x_values[model_name].append(x_max)
        if x_max < threshold: forecast_in_danger[model_name]=False
        if not forecast_in_danger[model_name] and x_max > threshold:
            forecast_reaction_times[model_name].append(timestep/120.0)
            forecast_in_danger[model_name]=True

# print(future_reaction_times)
# print(current_reaction_times)
# print(forecast_reaction_times)

# print("Future reaction times = ", np.array(future_reaction_times)- np.array(current_reaction_times))
# print("Forecast reaction times = ", np.array(forecast_reaction_times) - np.array(current_reaction_times))

plotting = True
if plotting:
    plot_folder = './plots/'
    plot_name = activity + f'{episode_num}_testing' + '.png'
    # Create the plot
    window_size = 15
    current_x_values_smooth = np.convolve(current_x_values, np.ones(window_size)/window_size, mode='same')
    future_x_values_smooth = np.convolve(future_x_values, np.ones(window_size)/window_size, mode='same')
    # plt.plot(time_values, current_x_values_smooth, label='Current', linestyle='--',zorder=9)
    current_mask = np.array(current_x_values_smooth) < 0.5
    plt.plot(time_values, np.where(current_mask, np.nan, current_x_values_smooth), label='Current', linestyle='--', zorder=9, alpha=1.0, color=color_map['Current'])
    plt.plot(time_values, np.where(current_mask, current_x_values_smooth, np.nan), label='Current', linestyle='--', zorder=9, alpha=0.25, color=color_map['Current'])
    


    # plt.plot(time_values, future_x_values_smooth, label='Future', linestyle='--',zorder=10)
    future_mask = np.array(future_x_values_smooth) < 0.5
    plt.plot(time_values, np.where(future_mask, np.nan, future_x_values_smooth), label='Future', linestyle='--', zorder=10, alpha=1.0, color=color_map['Future'])
    plt.plot(time_values, np.where(future_mask, future_x_values_smooth, np.nan), label='Future', linestyle='--', zorder=10, alpha=0.25, color=color_map['Future'])
    

    for i, (model_name, forecast_x) in enumerate(forecast_x_values.items()):
        window_size = 17
        # print(f'{model_name}: {forecast_x}')
        forecast_x_smooth = np.convolve(forecast_x, np.ones(window_size)/window_size, mode='same')

        mask = np.array(forecast_x_smooth) < 0.5
        plt.plot(time_values, np.where(mask, np.nan, forecast_x_smooth), label=f'{model_name}', linestyle='-', zorder=8-i, alpha=1.0, color=color_map[f'{model_name}'])
        plt.plot(time_values, np.where(mask, forecast_x_smooth, np.nan), label=f'{model_name}', linestyle='-', zorder=8-i, alpha=0.25, color=color_map[f'{model_name}'])
        

    plt.axhline(y=0.5, color='black', linestyle='-', linewidth=1,zorder=0,alpha=0.5)

    # Set plot title and labels
    plt.title('Wrist X-Position Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('X (m)')

    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = ['Future', 'FT-Mix-Trans-Wrist6', 'AMASS', 'Current']
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    legend_position = 'upper left'  # Position of the legend
    legend_bbox_to_anchor = (0, 1)  # Bbox coordinates of the legend
    legend_fontsize = 'small'  # Font size of the legend text

    plt.legend(unique_handles, unique_labels, loc=legend_position, bbox_to_anchor=legend_bbox_to_anchor, fontsize=legend_fontsize)

    plt.gcf().set_size_inches(10, 2)
    plt.subplots_adjust(bottom=0.25)  # Increase or decrease the values as needed


    # Save the plot as a PNG image file
    plt.savefig(plot_folder + plot_name, dpi=300)
    print(f'----- plotted to {plot_folder + plot_name} -----')

plt.clf()

plotting_mpjpe = True
if plotting_mpjpe:
    plot_folder = './plots_losses/'
    plot_name = activity + f'{episode_num}_testing' + '.png'
    # Create the plot
    for i, (model_name, forecast_loss) in enumerate(forecast_losses.items()):
        window_size = 21
        # print(f'{model_name}: {forecast_loss}')
        forecast_loss_smooth = np.convolve(forecast_loss, np.ones(window_size)/window_size, mode='same')
        plt.plot(time_values, forecast_loss_smooth, label=f'{model_name}', linestyle='-',zorder=8-i,color=color_map[f'{model_name}'])

    # Set plot title and labels
    plt.title('MPJPE Loss Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('MPJPE (mm)')

    # Add a legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    legend_position = 'upper left'  # Position of the legend
    legend_bbox_to_anchor = (0, 1)  # Bbox coordinates of the legend
    legend_fontsize = 'small'  # Font size of the legend text

    plt.legend(unique_handles, unique_labels, loc=legend_position, bbox_to_anchor=legend_bbox_to_anchor, fontsize=legend_fontsize)

    plt.gcf().set_size_inches(10, 2)
    plt.subplots_adjust(bottom=0.25)  # Increase or decrease the values as needed


    # Save the plot as a PNG image file
    plt.savefig(plot_folder + plot_name, dpi=300)
    print(f'----- plotted to {plot_folder + plot_name} -----')


# import pdb; pdb.set_trace()
