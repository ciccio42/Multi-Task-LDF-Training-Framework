import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
import cv2


def get_traj_data(finetuning_datasets_path, ur5_uni_paths):
    # search for 1 traj for each dataset
    actions_data = {}
    dataset_names = []
    traj_paths = []
    finetuning_datasets = os.listdir(finetuning_datasets_path)
    for dataset_name in finetuning_datasets:
        if 'converted' in dataset_name and not 'old' in dataset_name:
            dataset_path = finetuning_datasets_path + f'/{dataset_name}'
            for dir,subdirs,files in os.walk(dataset_path):
                found_pkl = False
                for file in files:
                    if '.pkl' in file and not 'task_embedding' in file:
                        traj_path = f'{dir}/{file}'    
                        traj_paths.append(traj_path)
                        
                        dataset_names.append(dataset_name)
                        actions_data[dataset_name] = {}
                        found_pkl = True
                        break
                if found_pkl:
                    break
    
    for folder_path in ur5_uni_paths:
        traj_path = f'{folder_path}/task_00/traj000.pkl'
        traj_paths.append(traj_path)
        
        dataset_names.append(folder_path.split('/')[-1])
        actions_data[folder_path.split('/')[-1]] = {}
        
    # open trajectories
    traj_datas = []
    for traj_path in traj_paths:
        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)
        traj_datas.append(traj_data)
        
    return traj_datas, dataset_names, actions_data


def fill_actions_data(traj_datas, actions_data):
        # save actions
    
    for i, traj_data in enumerate(traj_datas):
        if len(traj_data['traj'].get(1)['action']) == 7:
            print(f'{dataset_names[i]} has action len 7')
            key_dataset = dataset_names[i]
            actions_data[key_dataset]['x'] = []
            actions_data[key_dataset]['y'] = []
            actions_data[key_dataset]['z'] = []
            actions_data[key_dataset]['roll'] = []
            actions_data[key_dataset]['pitch'] = []
            actions_data[key_dataset]['yaw'] = []
            actions_data[key_dataset]['gripper'] = []
            
            traj = traj_data['traj']
            for step in range(traj_data['len']):
                try:
                    action = traj.get(step)['action']
                    actions_data[key_dataset]['x'].append(action[0])
                    actions_data[key_dataset]['y'].append(action[1])
                    actions_data[key_dataset]['z'].append(action[2])
                    actions_data[key_dataset]['roll'].append(action[3])
                    actions_data[key_dataset]['pitch'].append(action[4])
                    actions_data[key_dataset]['yaw'].append(action[5])
                    actions_data[key_dataset]['gripper'].append(action[6])
                except KeyError:
                    print(f'[WARNING] there is shift in {key_dataset}')
                    pass
        
        elif len(traj_data['traj'].get(1)['action']) == 8:
            print(f'{dataset_names[i]} has action len 8')
            key_dataset = dataset_names[i]
            actions_data[key_dataset]['x'] = []
            actions_data[key_dataset]['y'] = []
            actions_data[key_dataset]['z'] = []
            actions_data[key_dataset]['gripper'] = []
            actions_data[key_dataset]['q1'] = []
            actions_data[key_dataset]['q2'] = []
            actions_data[key_dataset]['q3'] = []
            actions_data[key_dataset]['q4'] = []
            
            traj = traj_data['traj']
            for step in range(traj_data['len']):
                try:
                    action = traj.get(step)['action']
                    actions_data[key_dataset]['x'].append(action[0])
                    actions_data[key_dataset]['y'].append(action[1])
                    actions_data[key_dataset]['z'].append(action[2])
                    actions_data[key_dataset]['q1'].append(action[3])
                    actions_data[key_dataset]['q2'].append(action[4])
                    actions_data[key_dataset]['q3'].append(action[5])
                    actions_data[key_dataset]['q4'].append(action[6])
                    actions_data[key_dataset]['gripper'].append(action[7])
                except KeyError:
                    print(f'[WARNING] there is shift in {key_dataset}')
                    pass

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--save_png_name", default='prova_x')
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    save_file_name = args.save_png_name + '.png'
    print(f'saving to {save_file_name}')
        
    finetuning_datasets_path = '/user/frosa/multi_task_lfd/datasets'
    real_ur5_dataset_path = '/raid/home/frosa_Loc/opt_dataset/pick_place/real_new_ur5e_pick_place'
    sim_ur5_dataset_path = '/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place'
    ur5_uni_paths = [real_ur5_dataset_path, sim_ur5_dataset_path]
    
    
    traj_datas, dataset_names, actions_data = get_traj_data(finetuning_datasets_path, ur5_uni_paths)
    

    
    fill_actions_data(traj_datas, actions_data)
        

            
            


            
            
    fig = plt.figure()
    fig.set_figheight(40.0)
    fig.set_figwidth(40.0)   
    # grid spec
    
    gs0 = gridspec.GridSpec(3,3, figure=fig)
    for i in range(len(traj_datas)):
        gs00 = gridspec.GridSpecFromSubplotSpec(3,3, subplot_spec=gs0[i])

        if len(actions_data[dataset_names[i]].keys()) == 7:
            ax01 = fig.add_subplot(gs00[0,0])
            ax01.title.set_text('x')
            ax02 = fig.add_subplot(gs00[0,1])   
            ax02.title.set_text('y')
            ax03 = fig.add_subplot(gs00[0,2])
            ax03.title.set_text('z')
            # ax02 = fig.add_subplot(gs00[-2,2], projection='polar')   
            # ax03 = fig.add_subplot(gs00[-2,3], projection='polar')   
            # ax04 = fig.add_subplot(gs00[-1,2], projection='polar')
            ax04 = fig.add_subplot(gs00[1,0])
            ax04.title.set_text('roll')
            ax05 = fig.add_subplot(gs00[1,1])
            ax05.title.set_text('pitch')
            ax06 = fig.add_subplot(gs00[1,2])
            ax06.title.set_text('yaw')
            ax07 = fig.add_subplot(gs00[2,1])
            ax07.title.set_text('gripper')
            
            
            key_dataset = dataset_names[i]
            ax01.plot(actions_data[key_dataset]['x'])
            ax02.plot(actions_data[key_dataset]['y'])
            ax03.plot(actions_data[key_dataset]['z'])
            
            ax04.plot(actions_data[key_dataset]['roll'])
            ax05.plot(actions_data[key_dataset]['pitch'])
            ax06.plot(actions_data[key_dataset]['yaw'])
            
            ax07.plot(actions_data[key_dataset]['gripper'])
            
        elif len(actions_data[dataset_names[i]].keys()) == 8:
            ax01 = fig.add_subplot(gs00[0,0])
            ax01.title.set_text('x')
            ax02 = fig.add_subplot(gs00[0,1])   
            ax02.title.set_text('y')
            ax03 = fig.add_subplot(gs00[0,2])
            ax03.title.set_text('z')
            # ax02 = fig.add_subplot(gs00[-2,2], projection='polar')   
            # ax03 = fig.add_subplot(gs00[-2,3], projection='polar')   
            # ax04 = fig.add_subplot(gs00[-1,2], projection='polar')
            ax04 = fig.add_subplot(gs00[1,0])
            ax04.title.set_text('q1')
            ax05 = fig.add_subplot(gs00[1,1])
            ax05.title.set_text('q2')
            ax06 = fig.add_subplot(gs00[1,2])
            ax06.title.set_text('q3')
            ax07 = fig.add_subplot(gs00[2,1])
            ax07.title.set_text('q4')
            ax08 = fig.add_subplot(gs00[2,2])
            ax08.title.set_text('gripper')
            
            key_dataset = dataset_names[i]
            ax01.plot(actions_data[key_dataset]['x'])
            ax02.plot(actions_data[key_dataset]['y'])
            ax03.plot(actions_data[key_dataset]['z'])
            
            ax04.plot(actions_data[key_dataset]['q1'])
            ax05.plot(actions_data[key_dataset]['q2'])
            ax06.plot(actions_data[key_dataset]['q3'])
            ax07.plot(actions_data[key_dataset]['q4'])
            
            ax08.plot(actions_data[key_dataset]['gripper'])
        
        
    plt.savefig(save_file_name)

    print('blalblablal')
    print('csdovnvsdnc')  
    
    
    
# index = 4
# for i in range(traj_datas[index]['len']):
#         if traj_datas[index]['traj'].get(i)['action'][3] > 1.0 or traj_datas[index]['traj'].get(i)['action'][3] < -1.0:
#             action = traj_datas[index]['traj'].get(i)['action'][3]
#             print(f'{action}, step: {i}')

# index = -2
# for i in range(traj_datas[index]['len']):
#     action = traj_datas[index]['traj'].get(i)['action'][3]
#     print(f'{action}, step: {i}')
        
    # draw
    
