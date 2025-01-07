import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes, RGBAxes
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--num_steps", default=1)
    parser.add_argument("--save_gif_name", default='prova_x')
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    num_steps = int(args.num_steps)
    save_file_name = args.save_gif_name + '.gif'
    save_mp4_file = args.save_gif_name + '.mp4'
    print(f'number of animations steps: {num_steps}')
    print(f'saving to {save_file_name}')
        
    finetuning_datasets_path = '/user/frosa/multi_task_lfd/datasets'
    real_ur5_dataset_path = '/raid/home/frosa_Loc/opt_dataset/pick_place/real_new_ur5e_pick_place'
    sim_ur5_dataset_path = '/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/ur5e_pick_place'
    ur5_uni_paths = [real_ur5_dataset_path, sim_ur5_dataset_path]
    
    
    # search for 1 traj for each dataset
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
                        found_pkl = True
                        break
                if found_pkl:
                    break
    
    for folder_path in ur5_uni_paths:
        traj_path = f'{folder_path}/task_14/traj002.pkl'
        traj_paths.append(traj_path)
        
    # open trajectories
    traj_datas = []
    for traj_path in traj_paths:
        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)
        traj_datas.append(traj_data)
                
    action_plots = 9
    image_plots = 9
    total_plots = image_plots + action_plots
    col_num = 3
    row_num, col_num = total_plots//col_num, col_num
    print('done loading, now plotting')
    
    # fig, axes = plt.subplots(row_num,col_num)
    
    fig = plt.figure()
    fig.set_figheight(65.0)
    fig.set_figwidth(65.0)    
    
    image_axes = []
    position_axes = []
    phi_axes = []
    theta_axes = []
    psi_axes = []
    x_position_2d_axes = []
    y_position_2d_axes = []
    z_position_2d_axes = []
    
    # take the minimum traj length
    min_len = np.inf
    for t in traj_datas:
        if t['len'] < min_len:
            min_len = t['len']
        
    old_num_steps = num_steps
    num_steps = num_steps if num_steps <= min_len else min_len
    print(f'min len: {min_len}. You specified {num_steps} steps. So plotting {num_steps}')
    
    #get actions
    traj_actions = [[[] for j in range(3)] for i in range(len(traj_datas))]
    for act_idx,action_list in enumerate(traj_actions):
        for step in range(num_steps):
            try:
                for el_idx, el in enumerate(action_list):
                    el.append(traj_datas[act_idx]['traj'].get(step)['action'][el_idx])
            except KeyError:
                step+=1
                for el_idx, el in enumerate(action_list):
                    el.append(traj_datas[act_idx]['traj'].get(step)['action'][el_idx])

        action_list = [np.array(i) for i in action_list]
        traj_actions[act_idx] = action_list
        
    #get rotations
    traj_rotations = [[[] for j in range(3)] for i in range(len(traj_datas))]
    for rot_idx,rotation_list in enumerate(traj_rotations): # datasets
        for step in range(num_steps):
            try:
                for el_idx, el in enumerate(rotation_list): # 3 angles
                    el.append(traj_datas[rot_idx]['traj'].get(step)['action'][el_idx+3])
            except KeyError:
                step+=1 
                for el_idx, el in enumerate(rotation_list):
                    el.append(traj_datas[rot_idx]['traj'].get(step)['action'][el_idx+3])
        
        rotation_list = [np.array(i) for i in rotation_list]
        traj_rotations[rot_idx] = rotation_list

    #grid specifications
    gs0 = gridspec.GridSpec(3,3, figure=fig)
    for i in range(9):
        gs00 = gridspec.GridSpecFromSubplotSpec(5,5, subplot_spec=gs0[i])

        ax0 = fig.add_subplot(gs00[:3,:])
        ax0.title.set_text('image')
        ax01 = fig.add_subplot(gs00[3:,0:2], projection='3d')
        ax01.title.set_text('3d pos plot')
        
        # set limits according to max and min value
        try:
            min_x, max_x = np.min(traj_actions[i][0]), np.max(traj_actions[i][0])
            min_y, max_y = np.min(traj_actions[i][1]), np.max(traj_actions[i][1])
            min_z, max_z = np.min(traj_actions[i][2]), np.max(traj_actions[i][2])
            ax01.axes.set_xlim3d(left=min_x , right=max_x)
            ax01.axes.set_ylim3d(bottom=min_y, top=max_y) 
            ax01.axes.set_zlim3d(bottom=min_z, top=max_z) 
        except IndexError:
            pass
        
            
        ax02 = fig.add_subplot(gs00[-2,2])
        ax02.title.set_text('roll')
        ax03 = fig.add_subplot(gs00[-2,3])   
        ax03.title.set_text('pitch')
        ax04 = fig.add_subplot(gs00[-2,4])
        ax04.title.set_text('yaw')
        # ax02 = fig.add_subplot(gs00[-2,2], projection='polar')   
        # ax03 = fig.add_subplot(gs00[-2,3], projection='polar')   
        # ax04 = fig.add_subplot(gs00[-1,2], projection='polar')
        ax05 = fig.add_subplot(gs00[-1,-3])
        ax05.title.set_text('x')
        ax06 = fig.add_subplot(gs00[-1,-2])
        ax06.title.set_text('y')
        ax07 = fig.add_subplot(gs00[-1,-1])
        ax07.title.set_text('z')
        
        
        image_axes.append(ax0)
        position_axes.append(ax01)
        phi_axes.append(ax02)
        theta_axes.append(ax03)
        psi_axes.append(ax04)
        angle_axes = [phi_axes, theta_axes, psi_axes]
        
        x_position_2d_axes.append(ax05)
        y_position_2d_axes.append(ax06)
        z_position_2d_axes.append(ax07)
        position_2d_axes = [x_position_2d_axes, y_position_2d_axes, z_position_2d_axes]
        
    pos_lines = [ax.plot([],[],[])[0] for ax in position_axes]

    
    def animate(i, actions_lists, pos_lines, rot_arrays, angle_axes, position_2d_axes):
        
        print(i)
        
        #------ image plot
        for idx,traj_dict in enumerate(traj_datas):
            
            image_dict = traj_dict['traj'].get(i)
            # image = np.moveaxis(obs.detach().cpu().numpy()*255, 0, -1)
            try:
                obs = image_dict['obs']['image']
            except KeyError:
                obs = image_dict['obs']['camera_front_image']
    
            if traj_dict['env_type'] == 'pick_place':
                obs = obs[:,:,::-1] #BGR->RGB
                
            ax_r, ax_g, ax_b = make_rgb_axes(image_axes[idx], pad=0.02)
            # kwargs = dict(origin="lower", interpolation="nearest")
            im_rgb, im_r, im_g, im_b = obs, obs[:,:,0], obs[:,:,1], obs[:,:,2]
            # ax[j].imshow(im_rgb, **kwargs)
            # ax_r.imshow(im_r, **kwargs)
            # ax_g.imshow(im_g, **kwargs)
            # ax_b.imshow(im_b, **kwargs)
            image_axes[idx].imshow(im_rgb)
                        
            color_mappable = ax_r.imshow(im_r, cmap='gray')
            # fig.colorbar(color_mappable, ax=ax[j])
            ax_g.imshow(im_g, cmap='gray')
            ax_b.imshow(im_b, cmap='gray')  
            
        #------ action plot
        
        # for line, action in zip(pos_lines, action_list):
        #     line.set_data_3d(action_list[:i])
        
        for idx, line in enumerate(pos_lines):
            try:
                plot_action_list = actions_lists[idx]
                line.set_data_3d(plot_action_list[0][:i],
                                        plot_action_list[1][:i],
                                        plot_action_list[2][:i])
            except IndexError:
                break
            
        #----- rotation plot
        for k, axes_list_for_angle in enumerate(angle_axes): # all axes for an angle
            for z, ax_rot in enumerate(axes_list_for_angle): # single ax for the angle
                try:
                    rot_arr = rot_arrays[z][k]
                    ax_rot.plot(rot_arr[:i])
                except IndexError:
                    break
                
        #----- 2d positions plot
        for k, axes_list_for_angle in enumerate(position_2d_axes): # all axes for an angle
            for z, ax_rot in enumerate(axes_list_for_angle): # single ax for the angle
                try:
                    act_arr = actions_lists[z][k]
                    ax_rot.plot(act_arr[:i])
                except IndexError:
                    break
        
        position_2d_axes
            
        return pos_lines    
    
    # save gif
    anim_fig = FuncAnimation(fig, animate, fargs=(traj_actions, pos_lines, traj_rotations, angle_axes, position_2d_axes), interval=500, frames=num_steps)
    anim_fig.save(save_file_name)
    # writervideo = FFMpegWriter(fps=10) 
    # anim_fig.save(save_mp4_file, writer=writervideo)
    
    print('end')
                
    
    