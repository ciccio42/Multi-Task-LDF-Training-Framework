

# input: .json file with the following structures
# output: .json with the same structure but with the couples
# number of combinations: assuming that for T task, one has N trajectories.
# if we want to make couples of 2 elements where order doesn't count,
# this is a dispositions with repetition D'n,k = n^2.
# So for a task of N=10 traj, we have 100 couples
# for a task of N=4 traj, we have 4^2=16 couples:
# T1:T1
# T1:T2
# T1:T3
# T1:T4
# T2:T1
# T2:T2
# T2:T3
# T2:T4
# T3:T1
# T3:T2
# T3:T3
# T3:T4
# T4:T1
# T4:T2
# T4:T3
# T4:T4


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--trajectory_json_file_path", default='val_pkl_paths.json')
        
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    
    import json
    with open(args.trajectory_json_file_path, 'r') as file:
        data = json.load(file)
        
    couples_dataset = {}
    for dataset_str in data.keys():
        couples_dataset[dataset_str] = {}
        for task_str in data[dataset_str].keys():
            if type(data[dataset_str][task_str]) == list: # if we found idxs
                couples_dataset[dataset_str][task_str] = []
                samples = data[dataset_str][task_str]
                for s_1 in samples:
                    for s_2 in samples:
                        couples_dataset[dataset_str][task_str].append((s_1, s_2))
            else:
                couples_dataset[dataset_str][task_str] = {}
                for subtask_str in data[dataset_str][task_str].keys():
                    if type(data[dataset_str][task_str][subtask_str]) == list: # if we found idxs
                        couples_dataset[dataset_str][task_str][subtask_str] = []
                        samples = data[dataset_str][task_str][subtask_str]
                        for s_1 in samples:
                            for s_2 in samples:
                                couples_dataset[dataset_str][task_str][subtask_str].append((s_1, s_2))
                    else:
                        raise NotImplementedError
    
    # import os
    # os.mkdir('traj_couples/')
    orig_json_name = args.trajectory_json_file_path.split('/')[-1].split('.')[0]
    with open(f"traj_couples/{orig_json_name}_couples.json", "w") as outfile: 
        json.dump(couples_dataset,outfile,indent=2)
    
    

