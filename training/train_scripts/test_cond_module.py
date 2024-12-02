from multi_task_il.datasets.command_encoder.cond_module import CondModule
import torch
from train_scripts.train_cond_module import get_train_val_loader
import seaborn as sns
import numpy as np
import cv2
from multi_task_il.datasets.command_encoder.multi_task_command_encoder import CommandEncoderDataset, CommandEncoderSampler
from torch.utils.data import DataLoader
import random


SEED = 1234
random.seed(SEED)

CUDA_DEVICE = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

#### cuda device
print(f"current device: {torch.cuda.current_device()}")
if CUDA_DEVICE != 0:
    print(f"switching to {CUDA_DEVICE}...")
    torch.cuda.set_device(CUDA_DEVICE)
    print(f"current device: {torch.cuda.current_device()}")

DATA_AUGS = {
            "old_aug": False,
            "brightness": [0.9, 1.1],
            "contrast": [0.9, 1.1],
            "saturation": [0.9, 1.1],
            "hue": [0.0, 0.0],
            "p": 0.1,
            "horizontal_flip_p": 0.1,
            "brightness_strong": [0.875, 1.125],
            "contrast_strong": [0.5, 1.5],
            "saturation_strong": [0.5, 1.5],
            "hue_strong": [-0.05, 0.05],
            "p_strong": 0.5,
            "horizontal_flip_p_strong": 0.5,
            "null_bb": False,
        }
DEBUG = False
DATASET = 'test' # use this for see how the model performs with train, val or test data
MODEL_PATH = '/user/frosa/multi_task_lfd/checkpoint_save_folder/1Task-pick_place-cond_module-Batch32/model_save-48.pt'
ROBOT = 'panda'

## loading model
cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True).to(device)
weights = torch.load(MODEL_PATH, weights_only=True)

assert DATASET == 'train' or DATASET == 'val' or DATASET == 'test'

cond_module.load_state_dict(weights)
cond_module.eval()

if DATASET == 'train':
    train_loader, _ = get_train_val_loader(DATA_AUGS, 16, robot=ROBOT, shuffle=False)
    loader = train_loader
elif DATASET == 'val':
    _, val_loader = get_train_val_loader(DATA_AUGS, 16, robot=ROBOT, shuffle=False)
    loader = val_loader
elif DATASET == 'test':
    test_dataset = CommandEncoderDataset(data_augs=DATA_AUGS,
                                         robot=ROBOT,
                                         mode='test',
                                         use_embedding_centroids=True,
                                         n_train_and_val_samples_per_subtask=60, # set the number of train/val samples that the model used for training/validation
                                         n_test_samples_per_subtask=30)
    
    test_sampler = CommandEncoderSampler(test_dataset, batch_size=16, shuffle=False)
    # with sampler
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    loader = test_loader
    
embedding_avg_subtask = {} # it contains the sum of the embedding of the subtask
embedding_avg_subtask_numpy = {} # it contains the sum of the embedding of the subtask

y = []
with torch.no_grad():
    for _i, data in enumerate(loader):
            video_input, sentence = data['demo_data']['demo'], data['embedding']['centroid_embedding']
            video_input = video_input.to(device)
            output_embedding = cond_module(video_input)
            y += [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            if DEBUG:
                if _i == 0:
                    for indx, sample in enumerate(video_input):
                        # i-th sample
                        for t in range(4):
                            if t == 3:
                                img_tensor = np.moveaxis(sample[t].detach().cpu().numpy()*255, 0, -1)
                                print(data['embedding']['sentence'])
                                cv2.imwrite(f"{indx}_frame_{t}.png", img_tensor)
            if _i != 0:
                embeddings_tensor = torch.cat((embeddings_tensor, output_embedding), 0)
                for _idx, emb in enumerate(output_embedding):
                    embedding_avg_subtask[_idx] = torch.cat((embedding_avg_subtask[_idx], emb.unsqueeze(0)), 0)
            else:
                embeddings_tensor = output_embedding
                for _idx, emb in enumerate(output_embedding):
                    embedding_avg_subtask[_idx] = emb.unsqueeze(0)       

for _idx, val in embedding_avg_subtask.items():
    embedding_avg_subtask[_idx] = torch.mean(embedding_avg_subtask[_idx], 0).detach().cpu()
    embedding_avg_subtask_numpy[_idx] = embedding_avg_subtask[_idx].numpy()
    if _idx == 0:
        embedding_avg_tensor = embedding_avg_subtask[_idx].unsqueeze(0)
    else:
        embedding_avg_tensor = torch.cat((embedding_avg_tensor, embedding_avg_subtask[_idx].unsqueeze(0)), 0)

print(cond_module)

import time
from IPython.display import display
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# create the dataframe for single embeddings
embeddings_tensor = embeddings_tensor.detach().cpu()
embedding_numpy = embeddings_tensor.numpy()
feat_cols = [ 'e'+str(i) for i in range(embeddings_tensor.shape[1]) ]
df = pd.DataFrame(embedding_numpy,columns=feat_cols)
df['y'] = y # label numerica
df['label'] = df['y'].apply(lambda i: str(i)) # label di tipo stringa

# # create dataframe for the centroids
df_centroids = pd.DataFrame.from_dict(embedding_avg_subtask_numpy, orient='index')
df_centroids['y'] = list(embedding_avg_subtask.keys())

union_tensor = torch.cat((embeddings_tensor, embedding_avg_tensor), 0)

assert torch.all(union_tensor[:-16] == embeddings_tensor).item()
assert torch.all(union_tensor[-16:] == embedding_avg_tensor).item()
# create TSNE object
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300, random_state=1) # vedere se cambiare parametri
# tsne_results = tsne.fit_transform(embeddings_tensor)
# tsne_results_centroids = tsne.fit_transform(embedding_avg_tensor)
tsne_results = tsne.fit_transform(union_tensor)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

## add columns to df

df['tsne-2d-one'] = tsne_results[:-16,0]
df['tsne-2d-two'] = tsne_results[:-16,1]

df_centroids['tsne-2d-one'] = tsne_results[-16:,0]
df_centroids['tsne-2d-two'] = tsne_results[-16:,1]

### plot tsne results
import colorcet as cc
palette = sns.color_palette(cc.glasbey, n_colors=16)

plt.figure(figsize=(16,10))
ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=palette,
    data=df,
    s=200,
    legend=False,
    # alpha=0.3
)

ax = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=palette,
    data=df_centroids,
    marker="*",
    s=400,
    legend="full",
    ax=ax
)

from datetime import datetime
print("saving results...")
ts = datetime.now().strftime("%m-%d_%H:%M")

# mode = 'train' if TEST_TRAIN_SET else 'val'
save_image_path = f'/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/train_scripts/img_{DATASET}_dataset_{ts}.png'
plt.savefig(save_image_path)
print("saved images to {}".format(save_image_path))






