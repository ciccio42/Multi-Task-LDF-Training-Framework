import glob
import os
import json

from multi_task_il.datasets import split_files


def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            return True

# ---------------------------------------------------README--------------------------------------------------------
# we want to create 2 files, to use for training and validate both cond_module and rt1_video_conditioned.
# We want to create them for 2 purposes:
# 1) We don't want that when rt1_video_conditioned is validated, the `cond_module` module of `rt1_video_conditioned`
# receives as input demostrations which has already seen in its previous training phase. This may happen if in the
# training of cond_module is adopted a split different to the one used in the training of `rt1_video_conditioned`.
# 2) We want to create a dataloader for pretraining on all the datasets we gather for our finetuning project.
# Having all the paths gathered by dataset, task is a good thing to do.
# ------------------------------------------------------------------------------------------------------------------

def main():
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', default='/user/frosa/multi_task_lfd/datasets')
    parser.add_argument('--panda_pick_place_folder', default='/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/panda_pick_place')
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--write_all_pkl_path", action='store_true', help="whether or not write all pkl paths")
    parser.add_argument("--write_train_pkl_path", action='store_true', help="whether or not write pkl for training file")
    parser.add_argument("--write_val_pkl_path", action='store_true', help="whether or not write pkl validation file")
    parser.add_argument("--split", default='0.9,0.1')
    parser.add_argument("--skip_pretraining_datasets", action='store_true')
    parser.add_argument("--ur5e_sim_panda", action='store_true')
    # parser.add_argument("--ur5e_real", action='store_true')
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    all_pkl_files_paths = {}
    train_pkl_files_paths = {}
    val_pkl_files_paths = {}
    pkl_files_paths = [all_pkl_files_paths, train_pkl_files_paths, val_pkl_files_paths]
    root_depth = 0
    train_val_split = [float(i) for i in args.split.split(',')]
    # ur5e_sim_pick_place_path = '/user/frosa/multi_task_lfd/ur_multitask_dataset/opt_dataset/pick_place/panda_pick_place'
    
    
    # this is the folder where we store the datasets for finetuning
    if not args.skip_pretraining_datasets:
        for root, dirs, files in os.walk(args.dataset_folder):
            
            if 'converted' in root or root == args.dataset_folder:    
                
                if root == args.dataset_folder:
                    root_depth = len(root.split('/')) #5
                    for dir in dirs:
                        if 'converted' in dir:
                            for pkl_dict in pkl_files_paths:
                                pkl_dict[dir] = {}
                else:
                    if len(dirs) != 0:
                        print(f"\n\n\n [{root}] \nscanning dirs: {dirs}")
                        
                        if len(root.split('/')) - root_depth == 1:
                            dataset_name = root.split('/')[-1]
                            for task_name in dirs:
                                if not folders_in(f'{root}/{task_name}'):
                                    for pkl_dict in pkl_files_paths:
                                        pkl_dict[dataset_name][task_name] = [] # list for pkls
                                else:
                                    for pkl_dict in pkl_files_paths:
                                        pkl_dict[dataset_name][task_name] = {} # dict for other subtasks
                        elif len(root.split('/')) - root_depth == 2:
                            dataset_name = root.split('/')[-2]
                            task_name = root.split('/')[-1]
                            for subtask_name in dirs:
                                if not folders_in(f'{root}/{subtask_name}'):
                                    for pkl_dict in pkl_files_paths:
                                        pkl_dict[dataset_name][task_name][subtask_name] = []
                                else:
                                    for pkl_dict in pkl_files_paths:
                                        pkl_dict[dataset_name][task_name][subtask_name] = {} # it should not happens
                                    print(f"***** WARNING ******")
                            
                                    
                    if len(files) != 0:
                        #---------------------------------------------
                        #TODO: dividire in train e split da qui dentro
                        #---------------------------------------------
                        print(f"\n\n\n [{root}] \nscanning files: {files}")
                        files = [i for i in files if i.endswith(".pkl") and i != 'task_embedding.pkl']
                        files = sorted(files)
                        if len(files) > 0:
                            if len(files) != 1:
                                idxs_train = split_files(len(files), train_val_split, 'train')
                                idxs_val = split_files(len(files), train_val_split, 'val')
                                for _idx, file in enumerate(files):
                                    # if '.pkl' in file:
                                    if len(root.split('/')) - root_depth == 3:
                                        dataset_name = root.split('/')[-3]
                                        task_name = root.split('/')[-2]
                                        subtask_name = root.split('/')[-1]
                                        all_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                        
                                        if _idx in idxs_train:
                                            train_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                        elif _idx in idxs_val:
                                            val_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                        
                                    else:
                                        dataset_name = root.split('/')[-2] 
                                        task_name = root.split('/')[-1]
                                        all_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                                
                                        if _idx in idxs_train:
                                            train_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                                        elif _idx in idxs_val:
                                            val_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                            else: # if we have only 1 trajectory, use for train and val
                                for _idx, file in enumerate(files):
                                    if len(root.split('/')) - root_depth == 3:
                                        dataset_name = root.split('/')[-3]
                                        task_name = root.split('/')[-2]
                                        subtask_name = root.split('/')[-1]
                                        
                                        for pkl_dict in pkl_files_paths:
                                            pkl_dict[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                                        
                                    else:
                                        dataset_name = root.split('/')[-2] 
                                        task_name = root.split('/')[-1]
                                        
                                        for pkl_dict in pkl_files_paths:
                                            pkl_dict[dataset_name][task_name].append(f'{root}/{file}')

    # this is to store pkl paths of the ur5e_panda_dataset
    if args.ur5e_sim_panda:
        print(f'searching into {args.panda_pick_place_folder}')
        for root, dirs, files in os.walk(args.panda_pick_place_folder):
            # print(root)
            # print(dirs)
            # print(files)
            # print('\n')
            if len(dirs) != 0:
                dataset_name = root.split('/')[-1]
                for pkl_dict in pkl_files_paths:
                    pkl_dict[dataset_name] = {}
                    
                for task in sorted(dirs):
                    for pkl_dict in pkl_files_paths:
                        pkl_dict[dataset_name][task] = []
            elif len(files) != 0:
                files = [i for i in files if i.endswith(".pkl") and i != 'task_embedding.pkl']
                files = sorted(files)
                idxs_train = split_files(len(files), train_val_split, 'train')
                idxs_val = split_files(len(files), train_val_split, 'val')
                task_name = root.split('/')[-1]
                for _idx, file in enumerate(files):
                    all_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                    if _idx in idxs_train:
                        train_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                    elif _idx in idxs_val:
                        val_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
        
    if args.write_all_pkl_path:
        with open("all_pkl_paths.json", "w") as outfile: 
            json.dump(all_pkl_files_paths,outfile,indent=2)
    if args.write_train_pkl_path:
        with open("train_pkl_paths.json", "w") as outfile: 
            json.dump(train_pkl_files_paths,outfile,indent=2)
    if args.write_val_pkl_path:
        with open("val_pkl_paths.json", "w") as outfile: 
            json.dump(val_pkl_files_paths,outfile,indent=2)        


if __name__ == '__main__':
    main()