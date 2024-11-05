import pickle as pkl
import os
import glob
import debugpy


PATH_TO_DATASET = "/home/rsofnc000/dataset/dagger_elaborated/pick_place/real_new_ur5e_pick_place"

if __name__ == '__main__':
    
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

    task_folders = glob.glob(os.path.join(PATH_TO_DATASET, 'task_*'))    
    
    for task_folder in task_folders:
        print(f"Task Folder {task_folder}")
        
        trj_paths = glob.glob(os.path.join(task_folder, '*.pkl'))
        
        for trj_path in trj_paths:
            print(trj_path)
            with open(trj_path, 'rb') as f:
                data = pkl.load(f)
                
            trj = data['traj']

            for t in range(len(trj)):
                if 'predicted_bb' in trj.get(t)['obs'].keys():
                    # print(trj.get(t)['obs']['predicted_bb'])
                    for camera_name in trj.get(t)['obs']['predicted_bb'].keys():
                        predicted_bbs_gpu = trj.get(t)['obs']['predicted_bb'][camera_name]
                        predicted_bbs_cpu = list()
                        for predicted_bb_gpu in predicted_bbs_gpu:
                            predicted_bbs_cpu.append(predicted_bb_gpu.cpu().detach().numpy())
                        
                        obs = trj.get(t)['obs']
                        obs['predicted_bb'][camera_name] = predicted_bbs_cpu
                        
                        trj.change_obs(t, obs)
                        
            
            pkl.dump({
                'traj': trj,
                'len': len(trj),
                'env_type': data['env_type'],
                'task_id': data['task_id']}, open(trj_path, 'wb'))