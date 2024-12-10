from sklearn.manifold import TSNE
import pandas as pd
import json
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from IPython.display import display

from multi_task_il.models.muse.muse import get_model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate_embeddings(model_torch, tokenize):
    embedding_subtasks = {}
    embedding_subtasks_tensor = {}
    ### PRODUZIONE EMBEDDING
    # qui chiediamo al sentence encoder di produrci un embedding per ogni frase a lui fornita
    for task in data.keys(): # per ogni task                
        embedding_subtasks[task] = {}
        embedding_subtasks_tensor[task] = {}
        for sub_task in data[task].keys(): # per ogni sottotask
            sub_task_idx = f'task_{int(sub_task):02d}'
            embedding_subtasks[task][sub_task_idx] = {}
            embedding_subtasks_tensor[task][sub_task_idx] = {}
            for (id,sentence) in enumerate(data[task][sub_task]): # (0, 'pick the... ')
                res = model_torch(tokenize(sentence))
                if id == 0 and sub_task == '0':
                    print(f"creating tensor:")
                    embeddings_tensor = torch.unsqueeze(res, 0)
                    res = torch.unsqueeze(res, 0)   
                else:
                    res = torch.unsqueeze(res, 0)
                    embeddings_tensor = torch.cat((embeddings_tensor, res))
                    
                embedding_subtasks[task][sub_task_idx][id] = {'sentence' : sentence, 'embedding' : res.detach().numpy()}
                embedding_subtasks_tensor[task][sub_task_idx][id] = {'sentence' : sentence, 'embedding' : res.detach()}
                
    return embedding_subtasks, embedding_subtasks_tensor, embeddings_tensor


def make_centroids(embedding_subtasks_tensor):
    centroids_subtasks = {}
    for _task in embedding_subtasks_tensor.keys():
        centroids_subtasks[_task] = {}
        for _sub_task in embedding_subtasks_tensor[_task].keys():
            _n_subtask_istances = len(embedding_subtasks_tensor[_task][_sub_task].keys())
            for _index in range(_n_subtask_istances):
                if _index == 0:
                    _subtask_tensor = embedding_subtasks_tensor[_task][_sub_task][_index]['embedding']
                    _sentence = embedding_subtasks_tensor[_task][_sub_task][_index]['sentence']
                else:
                    _subtask_tensor = torch.cat((_subtask_tensor, embedding_subtasks_tensor[_task][_sub_task][_index]['embedding']))
            
            _subtask_tensor = torch.mean(_subtask_tensor, 0)
            centroids_subtasks[_task][_sub_task] = {'sentence' : _sentence, 'centroid_embedding' : deepcopy(_subtask_tensor)}
            
    return centroids_subtasks

def save_embedding_centroids_per_subtask(centroids_subtasks, args):
    print(f"saving ONLY centroids...")
    import pickle
    # save embedding in .pkl files
    if not args.save_in_a_chosen_folder:
        print(f"saving the centroids in current dir...")
        root_file = 'centroids_commands_embs'
    else:
        assert args.save_embedding_path != '', 'you MUST specify a save path via argument --save_embedding_path [YOUR/SAVE_PATH]'
        print(f"saving at {args.save_embedding_path} ...")
        root_file = args.save_embedding_path
    if not os.path.exists(root_file):
        os.mkdir(root_file)
        
    for task in centroids_subtasks.keys():
        for subtask in centroids_subtasks[task].keys():
            if not os.path.exists(f'{root_file}/{subtask}'):
                os.mkdir(f'{root_file}/{subtask}')
            pickle.dump(centroids_subtasks[task][subtask], open(f'{root_file}/{subtask}/centroid_embedding.pkl', 'wb'))
            
def create_embedding_plot(embeddings_tensor):
    # create the dataframe
    embeddings_tensor = embeddings_tensor.detach()

    for _idx, _subtask in enumerate(centroids_subtasks['pick_place'].keys()):
        emb_centr = centroids_subtasks['pick_place'][_subtask]['centroid_embedding'].unsqueeze(0)
        embeddings_tensor = torch.cat((embeddings_tensor, emb_centr), 0)
        if _idx == 0:
            emb_centroids = emb_centr
        else:
            emb_centroids = torch.cat((emb_centroids, emb_centr), 0)

    y = []
    for i in range(16):
        y += ([str(i)] * 30)
    
    y_centr = []
    for i in range(16):
        y_centr += ([str(i)])

    embedding_numpy = embeddings_tensor.numpy()
    feat_cols = [ 'e'+str(i) for i in range(embeddings_tensor.shape[1]) ]
    df = pd.DataFrame(embedding_numpy[:-16],columns=feat_cols)
    df['y'] = y # label numerica
    df['label'] = df['y'].apply(lambda i: str(i)) # label di tipo stringa
    
    df_centr = pd.DataFrame(embedding_numpy[-16:],columns=feat_cols)
    df_centr['y'] = y_centr # label numerica
    df_centr['label'] = df_centr['y'].apply(lambda i: str(i)) # label di tipo stringa

    # create TSNE object
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300) # vedere se cambiare parametri
    tsne_results = tsne.fit_transform(embeddings_tensor)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    # add columns to df
    df['tsne-2d-one'] = tsne_results[:-16,0]
    df['tsne-2d-two'] = tsne_results[:-16,1]
    
    df_centr['tsne-2d-one'] = tsne_results[-16:,0]
    df_centr['tsne-2d-two'] = tsne_results[-16:,1]

    import colorcet as cc
    palette = sns.color_palette(cc.glasbey, n_colors=32)

    plt.figure(figsize=(16,10))
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y", # per ora non la uso visto che ogni campione è a se
        palette=palette,
        data=df,
        legend='full',
        # alpha=0.3
    )
    
    ax = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=palette,
        data=df_centr,
        marker="*",
        s=400,
        legend="full",
        ax=ax
    )
    
    from datetime import datetime
    ts = datetime.now().strftime("%m-%d_%H:%M")
    ## save clusters plot
    try:
        plt.savefig(f"centroid_figures/embeddings_clusters_{ts}.png")
    except Exception:
        os.mkdir("centroid_figures/")
        plt.savefig(f"centroid_figures/embeddings_clusters_{ts}.png")
            
def save(centroids_subtasks, args):
    if args.save_embeddings: # se vogliamo salvare TUTTI gli embeddings prodotti per ogni sottotask
        raise NotImplementedError
        # print(f"saving ALL embeddings")
        # import pickle
        # ### save embedding in .pkl files

        # if not SAVE_IN_OPT:
        #     print(f"saving in current dir...")
        #     root_file = 'commands_embs'
        # else:
        #     print(f"saving in opt...")
        #     root_file = '/raid/home/frosa_Loc/opt_dataset/pick_place/command_embs'
        # if not os.path.exists(root_file):
        #     os.mkdir(root_file)

        # for task in embedding_subtasks.keys():
        #     # if not os.path.exists(f'{root_file}/{task}'): # abbiamo solo pick_and_place
        #     #     os.mkdir(f'{root_file}/{task}')
        #     for subtask in embedding_subtasks[task].keys():
        #         # if not os.path.exists(f'{root_file}/{task}/{subtask}'):
        #         #     os.mkdir(f'{root_file}/{task}/{subtask}')
        #         if not os.path.exists(f'{root_file}/{subtask}'):
        #             os.mkdir(f'{root_file}/{subtask}')
        #         for _id, sentence_emb_dict in embedding_subtasks[task][subtask].items():
        #             # pickle.dump(sentence_emb_dict, open(f'{root_file}/{task}/{subtask}/embedding_{_id:03d}.pkl', 'wb'))
        #             pickle.dump(sentence_emb_dict, open(f'{root_file}/{subtask}/embedding_{_id:03d}.pkl', 'wb'))
    elif args.save_centroids: # se vogliamo salvare SOLO i centroidi degli embedding per ogni sottotask
        save_embedding_centroids_per_subtask(centroids_subtasks, args)

if __name__ == '__main__':    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--command_file_path', default='../training/multi_task_il/models/muse/commands/command_files_extended_11-01_21:01.json')
    parser.add_argument("--debug", action='store_true', help="whether or not attach the debugger")
    parser.add_argument("--save_df_as_pdf", action='store_true')
    parser.add_argument("--save_embeddings", action='store_true', help="whether or not saving embeddings produced by USE")
    parser.add_argument("--save_centroids", action='store_true', help="wheter or not saving embedding centroid for each subtask")
    parser.add_argument("--path_to_tokenizer", default='')
    parser.add_argument("--path_to_muse", default='')
    parser.add_argument("--show_results", action='store_true', help='whether or not visualize the embeddings produced')
    parser.add_argument("--read_from_json", action='store_true', help='use this if the commands are written in a json file')
    parser.add_argument("--save_in_a_chosen_folder", action='store_true', help='use this if you want to store in opt_dataset/ folder')
    parser.add_argument("--save_embedding_path", default='')
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    extended_json = True if 'extended' in args.command_file_path else False
    SAVE_IN_OPT = False # flag per salvare in opt_dataset/
    
    # get model and tokenizer
    model_torch, tokenize = get_model(args.path_to_muse, args.path_to_tokenizer)

    if args.read_from_json: # if commands are in a json file
        with open(args.command_file_path, 'r') as file:
            data = json.load(file)
        pick_place_commands = data['pick_place']
        number_of_task = len(pick_place_commands.keys())
    
    embedding_subtasks, embedding_subtasks_tensor, embeddings_tensor = generate_embeddings(model_torch, tokenize)    

    if args.save_centroids:
        centroids_subtasks = make_centroids(embedding_subtasks_tensor)

    save(centroids_subtasks, args) # save embedding centroids

    create_embedding_plot(embeddings_tensor)
    
    
    
