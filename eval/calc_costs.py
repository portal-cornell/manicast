import rospy
import json
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import numpy as np
from model import *
import torch
from utils.parser import args
from pynput import keyboard

def get_relevant_joints(all_joints, relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']):                       
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

def get_forecast(history_joints):
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

# def get_marker(id, pose, edge, ns = 'current', alpha=1, color=1):
#     marker = Marker()
#     marker.header.frame_id = "mocap"
#     marker.header.stamp = rospy.Time.now()
#     marker.type = marker.LINE_LIST
#     marker.id = id
#     marker.scale.x = 0.005
#     marker.action = marker.ADD 
#     marker.color.b = color
#     marker.color.a = alpha
#     marker.ns = f'{ns}-{relevant_joints[edge[0]]}_{relevant_joints[edge[1]]}'
#     pos1, pos2 = pose[edge[0]], pose[edge[1]]
#     p1, p2 = Point(), Point()
#     x, y, z = pos1.tolist()
#     p1.x, p1.y, p1.z = -x, z, y
#     x, y, z = pos2.tolist()
#     p2.x, p2.y, p2.z = -x, z, y
#     marker.points = [p1, p2]
#     return marker

def get_marker(id, pose, edge, ns = 'current', alpha=1, red=1, green=1, blue=1):
    relevant_joints=['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
    SCALE = 0.015
    marker = Marker()
    marker.header.frame_id = "mocap"
    marker.header.stamp = rospy.Time.now()
    marker.type = marker.LINE_LIST
    marker.id = id
    marker.scale.x = SCALE
    marker.action = marker.ADD 
    marker.ns = f'{ns}-{relevant_joints[edge[0]]}_{relevant_joints[edge[1]]}'
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = alpha
    p1m = Marker()
    p1m.header.frame_id = "mocap"
    p1m.header.stamp = rospy.Time.now()
    p1m.type = marker.SPHERE_LIST
    p1m.id = id + 101
    p1m.scale.x = SCALE*2
    p1m.scale.y = SCALE*2
    p1m.scale.z = SCALE*2
    p1m.action = p1m.ADD
    p1m.color.r = red
    p1m.color.g = green
    p1m.color.b = blue
    p1m.color.a = alpha/2
    pos1, pos2 = pose[edge[0]], pose[edge[1]]
    p1, p2 = Point(), Point()
    x, y, z = pos1.tolist()
    p1.x, p1.y, p1.z = -x, z, y
    x, y, z = pos2.tolist()
    p2.x, p2.y, p2.z = -x, z, y

    p1m.points = [p1, p2]
    marker.points = [p1, p2]
    return marker,p1m


def get_marker_array(current_joints, person = "Kushal"):
    id_offset = 100000 if person == 'Kushal' else 0
    color = 1 if person == "Kushal" else 0
    marker_array = MarkerArray()
    edges = [
            (0, 1), (0, 2),
            (1, 3), (3, 5),
            (2, 4), (4, 6)
        ]
    # extra edges to connect the pose back to the hips
    extra_edges = [(1, 7), (7, 8), (8, 2)]
    for idx, edge in enumerate(edges + extra_edges):
        tup = get_marker(idx+id_offset, current_joints, edge,ns=f'current', alpha=1, 
                         red=color, 
                         green=0.1, 
                         blue=0.0)
        marker_array.markers.append(tup[0])
        marker_array.markers.append(tup[1])

    return marker_array



if __name__ == '__main__':
    rospy.init_node('forecaster', anonymous=True)
    human_forecast = rospy.Publisher("/human_forecast", MarkerArray, queue_size=1)

    episode_file = f"/home/portal/MHAD_Processing/{args.activity}_data/{args.activity}_{args.ep_num}.json"
    # episode_file = "/home/portal/MHAD_Processing/final_handover_000.json"
    mapping_file = "mapping.json"
    with open(episode_file, 'r') as f:
        data = json.load(f)
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    relevant_joints = ['BackTop', 'LShoulderBack', 'RShoulderBack',
                        'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut', 'WaistLBack', 'WaistRBack']
    
    joint_used = np.array([mapping[joint_name] for joint_name in relevant_joints])
    
    pause = False
    def on_press(key):
        if key == keyboard.Key.space:
            pause = True
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    rate = rospy.Rate(1200)

    person_data = {}
    for stream_person in data:
        person_data[stream_person] = np.array(data[stream_person])

    
    def calc_cost(reaching_human, stirring_human):
        cost = 0
        reaching_human_wrist = reaching_human[6][0]
        stirring_human_wrist = stirring_human[6][0]
        print("Reaching Human = ", reaching_human[6])
        print("Stirring Human = ", stirring_human[6])

        if reaching_human_wrist > 0.4:
            print("why")
            cost = (0.96 - stirring_human_wrist)**2
        print("Cost = ", cost)
        # import pdb; pdb.set_trace()
        return cost
    costs = []
    duration = len(data[list(data.keys())[0]])
    for timestep in range(duration):
        print(round(timestep/120, 1))
        if not pause and listener.running:
            current_data = {}
            for stream_person in data:
                # if stream_person != "Kushal": continue
                joint_data = person_data[stream_person]
                current_joints = get_relevant_joints(joint_data[timestep])
                marker_array = get_marker_array(current_joints=current_joints,
                                                person=stream_person)
                current_data[stream_person] = np.array(current_joints)
                # future_markers = get_future_markers(future_joints)
                human_forecast.publish(marker_array)
                rate.sleep()
            cost = calc_cost(current_data['Kushal'], current_data['Prithwish'])
            # import pdb; pdb.set_trace()
            costs.append(cost)
        else:
            input("Press enter to continue")
            pause = False
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
    import matplotlib.pyplot as plt
    
    plt.plot(np.arange(duration)/120.0, costs)
    plt.savefig("costs.png", dpi=900)


# import pdb; pdb.set_trace()