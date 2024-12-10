import glob
import os
import json




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
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
        
    all_pkl_files_paths = {}
    root_depth = 0
    
    for root, dirs, files in os.walk(args.dataset_folder):
        
        if 'converted' in root or root == args.dataset_folder:    
            
            if root == args.dataset_folder:
                root_depth = len(root.split('/')) #5
                for dir in dirs:
                    if 'converted' in dir:
                        all_pkl_files_paths[dir] = {}
            else:
                if len(dirs) != 0:
                    print(f"\n\n\n [{root}] \nscanning dirs: {dirs}")
                    
                    if len(root.split('/')) - root_depth == 1:
                        dataset_name = root.split('/')[-1]
                        for task_name in dirs:
                            if not folders_in(f'{root}/{task_name}'):
                                all_pkl_files_paths[dataset_name][task_name] = [] # list for pkls
                            else:
                                all_pkl_files_paths[dataset_name][task_name] = {} # dict for other subtasks
                    elif len(root.split('/')) - root_depth == 2:
                        dataset_name = root.split('/')[-2]
                        task_name = root.split('/')[-1]
                        for subtask_name in dirs:
                            if not folders_in(f'{root}/{subtask_name}'):
                                all_pkl_files_paths[dataset_name][task_name][subtask_name] = []
                            else:
                                all_pkl_files_paths[dataset_name][task_name][subtask_name] = {} # it should not happens
                                print(f"***** WARNING ******")
                        
                                
                if len(files) != 0:
                    print(f"\n\n\n [{root}] \nscanning files: {files}")
                    for file in files:
                        if '.pkl' in file:
                            if len(root.split('/')) - root_depth == 3:
                                dataset_name = root.split('/')[-3] 
                                task_name = root.split('/')[-2]
                                subtask_name = root.split('/')[-1]
                                all_pkl_files_paths[dataset_name][task_name][subtask_name].append(f'{root}/{file}')
                            else:
                                dataset_name = root.split('/')[-2] 
                                task_name = root.split('/')[-1]
                                all_pkl_files_paths[dataset_name][task_name].append(f'{root}/{file}')
                    
            
    with open("all_pkl_paths.json", "w") as outfile: 
        json.dump(all_pkl_files_paths,outfile,indent=2)
        


if __name__ == '__main__':
    main()