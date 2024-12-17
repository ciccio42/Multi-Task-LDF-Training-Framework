import json
from multi_task_il.models.muse.muse import get_model
import pickle


def create_emb_and_save_pickle(traj_path, model_torch, tokenize):
    ''' query USE to produce a 512 embedding from the command
        associated to the task at traj_path
    '''
    print(f"computing embedding for {traj_path}...")
    with open(traj_path, "rb") as f:
        traj_data = pickle.load(f)
    command = traj_data['command']
    print(f"command: {command}")
    command_emb = model_torch(tokenize(command)).detach().numpy()
    save_path_command_emb = '/'.join(traj_path.split('/')[:-1])+'/task_embedding.pkl'
    pickle.dump(command_emb, open(save_path_command_emb, 'wb'))
    print(f"saved embedding at {save_path_command_emb}")
    return save_path_command_emb


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', default='./all_pkl_paths.json')
    parser.add_argument("--debug", default=False, help="whether or not attach the debugger")
    parser.add_argument("--path_to_tokenizer", default='')
    parser.add_argument("--path_to_muse", default='')
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    
    with open(args.task_json, 'r') as file:
        data = json.load(file)
    
    print(f"getting USE and tokenizer...")
    model_torch, tokenize = get_model(args.path_to_muse, args.path_to_tokenizer)
    print(f"done.")
    
    #----------------------- TODO -------------------------------------------
    #  1)usare task_count_info.json e fornire questa all'USE
    #  2)una volta prodotto l'embedding, combinarlo con il gruppo di comandi
    #  3)OPPURE creare un file separato in cui c'è corrispondenza con i task
    #------------------------------------------------------------------------
    
    embeddings_data = {}
    for dataset_name in data.keys():
        embeddings_data[dataset_name] = {}
        for task in data[dataset_name].keys():
            if type(data[dataset_name][task]) == list:
                embeddings_data[dataset_name][task] = []
                # USE for the embedding and save into embeddings_data
                istances_list = data[dataset_name][task]
                traj_path = istances_list[0] # take only one traj, the command string is the same for all elements in the folder
                # if traj_path == '/user/frosa/multi_task_lfd/datasets/taco_play_converted/stack_yellow_on_/taco_play_03201.pkl':
                #     continue
                save_path_command_emb = create_emb_and_save_pickle(traj_path, model_torch, tokenize)
                embeddings_data[dataset_name][task].append(save_path_command_emb)
                
            elif type(data[dataset_name][task]) == dict:
                embeddings_data[dataset_name][task] = {}
                for subtask in data[dataset_name][task].keys():
                    assert type(data[dataset_name][task][subtask]), f'error, data is of type {type(data[dataset_name][task][subtask])}'
                    embeddings_data[dataset_name][task][subtask] = []
                    istances_list = data[dataset_name][task][subtask]
                    traj_path = istances_list[0] # take only one traj
                    save_path_command_emb = create_emb_and_save_pickle(traj_path, model_torch, tokenize)
                    embeddings_data[dataset_name][task][subtask].append(save_path_command_emb)
                    
                    # USE for the embedding and save into embeddings_data  
                
    #----------------------- TODO ---------------------------
    #  visualizzare nello spazio gli embedding
    #-------------------------------------------------------             
            
    with open("embeddings_data.json", "w") as outfile: 
        json.dump(embeddings_data,outfile,indent=2) 
    
    # if args.asu:
    #     pass
    

    print('sdcdicod')
    print('blablablabla')
    
    