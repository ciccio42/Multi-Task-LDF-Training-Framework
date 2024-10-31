from sklearn.manifold import TSNE
import pandas as pd
import json
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import os

from IPython.display import display

from multi_task_il.models.muse.muse import get_model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# load data
# os.chdir('repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/muse/')
os.chdir('training/multi_task_il/models/muse/')
COMMAND_FILE_PATH = 'commands/commands_extended3.json'
EXTENDED_JSON = True if 'extended' in COMMAND_FILE_PATH else False
SAVE_DF_AS_PDF = False
SAVE_EMBEDDINGS = False
SAVE_EMBEDDINGS_ONLY_CENTROIDS = True
SAVE_IN_OPT = True # flag per salvare in opt_dataset/

print(f"current path: {os.getcwd()}")

# Open and read the JSON file
with open(COMMAND_FILE_PATH, 'r') as file:
    data = json.load(file)
    
# convert the string in embeddings
model_torch, tokenize = get_model()

pick_place_commands = data['pick_place']
NUMBER_OF_VARIATIONS = len(pick_place_commands['0'])
NUMBER_OF_TASK = len(pick_place_commands.keys())
TASK_NAMES = []

for task in data.keys():
    TASK_NAMES.append(task)

# embedding_pick_place_commands = {}

### LETTURA EMBEDDING DA FILE
if EXTENDED_JSON: # gli extended hanno una struttura diversa
    temp_dict = {}
    counter = 0
    y = []
    command_str = []
    for k,v in pick_place_commands.items():
        for el in v: # è una lista di comandi diversi
            temp_dict[str(counter)] = el
            counter+=1
            y.append(str(k))
            command_str.append(str(el))

    pick_place_commands = temp_dict

# print(f"COMMANDS: {pick_place_commands}")

embedding_subtasks = {}
embedding_subtasks_tensor = {}
for task in TASK_NAMES:
    embedding_subtasks[task] = {}
    embedding_subtasks_tensor[task] = {}

### PRODUZIONE EMBEDDING
# qui chiediamo al sentence encoder di produrci un embedding per ogni frase a lui fornita
for task in data.keys(): # per ogni task
    for sub_task in data[task].keys(): # per ogni sottotask
        
        sub_task_idx = f'task_{int(sub_task):02d}'
        embedding_subtasks[task][sub_task_idx] = {}
        embedding_subtasks_tensor[task][sub_task_idx] = {}
        
        for (id,sentence) in enumerate(data[task][sub_task]): # (0, 'pick the... ')
            # print(f"{k} -> {v}")

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

# tensore (N_elementi, size_embedding)  -> (16, 512)
print(f"resulting tensor shape: {embeddings_tensor.shape}")


if SAVE_EMBEDDINGS_ONLY_CENTROIDS:
    centroids_subtasks = {}
    centroids_subtasks[task] = {}

    _n_subtask_istances = len(embedding_subtasks_tensor[task][sub_task_idx].keys())

    for _task in embedding_subtasks_tensor.keys():
        for _sub_task in embedding_subtasks_tensor[_task].keys():
            for _index in range(_n_subtask_istances):
                if _index == 0:
                    _subtask_tensor = embedding_subtasks_tensor[_task][_sub_task][_index]['embedding']
                else:
                    _subtask_tensor = torch.cat((_subtask_tensor, embedding_subtasks_tensor[_task][_sub_task][_index]['embedding']))
            
            _subtask_tensor = torch.mean(_subtask_tensor, 0)
            centroids_subtasks[_task][_sub_task] = _subtask_tensor
        

# ## dataframe
# embeddings_tensor = embeddings_tensor.detach()
# embedding_numpy = embeddings_tensor.numpy()
# feat_cols = [ 'e'+str(i) for i in range(embeddings_tensor.shape[1]) ]
# df = pd.DataFrame(embedding_numpy,columns=feat_cols)
# df['y'] = y # label numerica
# df['label'] = df['y'].apply(lambda i: str(i)) # label di tipo stringas
# df['command_str'] = command_str

# print('Size of the dataframe: {}'.format(df.shape))
# display(df)





if SAVE_EMBEDDINGS: # se vogliamo salvare TUTTI gli embeddings prodotti per ogni sottotask
    print(f"saving ALL embeddings")
    import pickle
    ### save embedding in .pkl files

    if not SAVE_IN_OPT:
        print(f"saving in current dir...")
        root_file = 'commands_embs'
    else:
        print(f"saving in opt...")
        root_file = '/raid/home/frosa_Loc/opt_dataset/pick_place/command_embs'
    if not os.path.exists(root_file):
        os.mkdir(root_file)

    for task in embedding_subtasks.keys():
        # if not os.path.exists(f'{root_file}/{task}'): # abbiamo solo pick_and_place
        #     os.mkdir(f'{root_file}/{task}')
        for subtask in embedding_subtasks[task].keys():
            # if not os.path.exists(f'{root_file}/{task}/{subtask}'):
            #     os.mkdir(f'{root_file}/{task}/{subtask}')
            if not os.path.exists(f'{root_file}/{subtask}'):
                os.mkdir(f'{root_file}/{subtask}')
            for _id, sentence_emb_dict in embedding_subtasks[task][subtask].items():
                # pickle.dump(sentence_emb_dict, open(f'{root_file}/{task}/{subtask}/embedding_{_id:03d}.pkl', 'wb'))
                pickle.dump(sentence_emb_dict, open(f'{root_file}/{subtask}/embedding_{_id:03d}.pkl', 'wb'))
elif SAVE_EMBEDDINGS_ONLY_CENTROIDS: # se vogliamo salvare SOLO i centoidi degli embedding per ogni sottotask
    print(f"saving ONLY centroids...")
    import pickle
    ### save embedding in .pkl files

    if not SAVE_IN_OPT:
        print(f"saving in current dir...")
        root_file = 'centroids_commands_embs'
    else:
        print(f"saving in opt...")
        root_file = '/raid/home/frosa_Loc/opt_dataset/pick_place/centroids_commands_embs'
    if not os.path.exists(root_file):
        os.mkdir(root_file)

    for task in centroids_subtasks.keys():
        # if not os.path.exists(f'{root_file}/{task}'): # abbiamo solo pick_and_place
        #     os.mkdir(f'{root_file}/{task}')
        for subtask in centroids_subtasks[task].keys():
            # if not os.path.exists(f'{root_file}/{task}/{subtask}'):
            #     os.mkdir(f'{root_file}/{task}/{subtask}')
            if not os.path.exists(f'{root_file}/{subtask}'):
                os.mkdir(f'{root_file}/{subtask}')
            for i in range(len(embedding_subtasks[task][subtask].keys())): # il pezzone
                pickle.dump(centroids_subtasks[task][subtask], open(f'{root_file}/{subtask}/centroid_embedding_{i}.pkl', 'wb'))            
            

# ### EMBEDDING VISUALIZATION
# # create TSNE object
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300) # vedere se cambiare parametri
# tsne_results = tsne.fit_transform(embeddings_tensor)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# ## add columns to df
# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]

# # df[(df["y"] == '1') | (df["y"] == '0')]

# x = df[(df["y"] == '1')]["tsne-2d-one"].mean()
# y = df[(df["y"] == '1')]["tsne-2d-two"].mean()


# # calcolo media centroidi e metto in un dataframe
# indexes = []
# for i in range(NUMBER_OF_TASK):
#     x = df[(df["y"] == str(i))]["tsne-2d-one"].mean() # x coord del cluster
#     y = df[(df["y"] == str(i))]["tsne-2d-two"].mean() # y coord del cluster
#     if i == 0:
#         embeddings_tensor = torch.unsqueeze(res, 0)
#         cluster_tensor = torch.unsqueeze(torch.tensor((x,y)), 0)
#     else:
#         to_add = torch.unsqueeze(torch.tensor((x,y)), 0)
#         cluster_tensor = torch.cat((cluster_tensor, to_add))
#     indexes.append(i)

# cluster_tensor_numpy = cluster_tensor.detach().numpy()
# cluster_df = pd.DataFrame(cluster_tensor_numpy,columns=['x', 'y'])
# cluster_df['task'] = indexes


# ## dimensionality reduction
# plt.figure(figsize=(16,10))
# ax = sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y", # per ora non la uso visto che ogni campione è a se
#     palette=sns.color_palette("Spectral", NUMBER_OF_TASK),
#     data=df,
#     legend=False,
#     # alpha=0.3
# )

# ax = sns.scatterplot(
#     x="x", y="y",
#     hue="task",
#     palette=sns.color_palette("Spectral", NUMBER_OF_TASK),
#     data=cluster_df,
#     marker="*",
#     s=600,
#     legend="full",
#     ax=ax
# )
# # plt.show()

# from datetime import datetime

# if SAVE_IN_OPT: # salvo in opt_dataset
#     try:
#         root_dir = '/raid/home/frosa_Loc/opt_dataset/pick_place/command_embs'
#         plt.savefig(f"{root_dir}/visual_representation/embeddings_clusters.png")
#     except Exception:
#         os.mkdir(f'{root_dir}/visual_representation/')
#         root_dir = '/raid/home/frosa_Loc/opt_dataset/pick_place/command_embs'
#         plt.savefig(f"{root_dir}/visual_representation/embeddings_clusters.png") 
# else:
#     ts = datetime.now().strftime("%m-%d_%H:%M")
#     ## save clusters plot
#     try:
#         plt.savefig(f"figures/embeddings_clusters_{ts}.png")
#     except Exception:
#         os.mkdir("figures/")
#         plt.savefig(f"figures/embeddings_clusters_{ts}.png")
    

# ## save dataframe to json format
# try:
#     df.to_json(f'datasets/embeddings_dataset_{ts}.json')
# except Exception:
#     os.mkdir("datasets/")
#     df.to_json(f'datasets/embeddings_dataset_{ts}.json')






# if SAVE_DF_AS_PDF:
#     #https://stackoverflow.com/questions/32137396/how-do-i-plot-only-a-table-in-matplotlib
#     fig, ax =plt.subplots(figsize=(12,4))
#     ax.axis('tight')
#     ax.axis('off')
#     the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='center')

#     #https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
#     pp = PdfPages("embeddings_dataframe.pdf")
#     pp.savefig(fig, bbox_inches='tight')
#     pp.close()