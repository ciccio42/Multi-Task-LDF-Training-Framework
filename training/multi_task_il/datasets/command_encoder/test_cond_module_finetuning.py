
from multi_task_il.datasets.command_encoder.command_encoder_dataset import CommandEncoderFinetuningDataset, FinetuningCommandEncoderSampler
from multi_task_il.datasets.command_encoder.cond_module import CondModule
from multi_task_il.datasets.utils import collate_by_task
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import os
from copy import deepcopy


DATA_AUGS = {
            "old_aug": False,
            "brightness": [0.875, 1.125],
            "contrast": [0.5, 1.5],
            "saturation": [0.5, 1.5],
            "hue": [-0.05, 0.05],
            "p": 0.5,
            "horizontal_flip_p": 0.1,
            "brightness_strong": [0.875, 1.125],
            "contrast_strong": [0.5, 1.5],
            "saturation_strong": [0.5, 1.5],
            "hue_strong": [-0.05, 0.05],
            "p_strong": 0.5,
            "horizontal_flip_p_strong": 0.5,
            "null_bb": False,
        }

def create_val_loader(tasks_spec, black_list, data_augs):
    val_dataset = CommandEncoderFinetuningDataset(mode='val',
                                                tasks_spec=tasks_spec,
                                                jsons_folder='/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/bashes',
                                                black_list=black_list,
                                                data_augs=DATA_AUGS)

    samplerClass = FinetuningCommandEncoderSampler
    val_sampler = samplerClass(val_dataset,
                                shuffle=False)

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=20,
        worker_init_fn=lambda w: np.random.seed(
            np.random.randint(2 ** 29) + w),
        collate_fn=collate_by_task,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return val_loader

def list_of_strings(arg):
    return arg.split(',')

def make_centroids(embedding_subtasks_tensor, all_sentences):
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


def create_embedding_plot(embeddings_tensor, sentences, centroids):
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
            


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', default='/user/frosa/multi_task_lfd/checkpoint_save_folder/test_2_pretraining_5_datasets_no_droid_4_augmentations_10_epoch-Batch32/model_save-2530.pt')
    parser.add_argument('--cuda_device', default=1)
    parser.add_argument("--black_list", type=list_of_strings, default=['droid_converted', 'droid_converted_old'], help="datasets to exclude")
    parser.add_argument("--debug", default=False, help="whether or not attach the debugger")
    
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # cuda device
    print(f"current device: {torch.cuda.current_device()}")
    if args.cuda_device != 0:
        print(f"switching to {args.cuda_device}...")
        torch.cuda.set_device(int(args.cuda_device))
        print(f"current device: {torch.cuda.current_device()}")
    
    tasks_spec = [
        {
            "name": "pick_place",
            "n_tasks": 16,
            "crop": [20, 25, 80, 75],
            "n_per_task": 2,
            "task_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "skip_ids": [],
            "loss_mul": 1,
            "task_per_batch": 16,
            "traj_per_subtask": 100,
            "demo_per_subtask": 100,
        }
    ]

    val_loader = create_val_loader(tasks_spec, args.black_list, DATA_AUGS)

    ## loading model
    cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True).to(device)
    weights = torch.load(args.weights_path, weights_only=True)

    cond_module.load_state_dict(weights)
    cond_module.eval()

    model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(cond_module)
    print('Total params in cond module before freezing:', params)

    # freeze cond module
    for p in cond_module.parameters():
        p.requires_grad = False
        
    model_parameters = filter(lambda p: p.requires_grad, cond_module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print(cond_module)
    print('Total params in cond module after freezing:', params)

    embedding_dict = {} # store embeddings for each task
    # batch_count = 0
    for batch_idx, inputs in enumerate(val_loader):
        # folder_path = f'test_cond_module_example_batch_{batch_count}_cond_module'
        # for k in range(inputs['finetuning']['demo_data']['demo'].shape[0]):
        #     for i in range(4):
        #         image = inputs['finetuning']['demo_data']['demo'][k][i]
        #         sentence = inputs['finetuning']['sentence'][k]
        #         if not os.path.exists(folder_path):
        #             os.mkdir(folder_path)
        #         image = cv2.putText(np.ascontiguousarray(np.moveaxis(image.numpy()*255, 0, -1)),
        #                             sentence,
        #                             (2,10),
        #                             cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.3,
        #                             (255,0,0),
        #                             1,
        #                             cv2.LINE_AA)
        #         cv2.imwrite(f"{folder_path}/{k}_{i}.png", image)
        
        # batch_count+=1
        demos = inputs['finetuning']['demo_data']['demo'].to(device)
        batch_sentences = inputs['finetuning']['sentence']
        batch_output = cond_module(demos)
        if batch_idx == 0:
            # all_output = batch_output
            # all_sentences = batch_sentences            
            for sentence_idx, sentence in enumerate(batch_sentences):
                embedding_dict[sentence] = [] # create the list to store embeddings
                embedding_dict[sentence].append(batch_output[sentence_idx].detach().cpu().numpy())
        else:
            # all_output = torch.cat((all_output, batch_output), 0)
            # all_sentences.extend(batch_sentences)
            for sentence_idx, sentence in enumerate(batch_sentences):
                embedding_dict[sentence].append(batch_output[sentence_idx].detach().cpu().numpy())
            
        with open('test_sentences.txt', 'w') as f:
            for line in batch_sentences:
                f.write(f"{line}\n")
            f.write(f"\n\n***")
        
        
        # print(inputs['sentence'])    
    
    print('end')
    # centroids = make_centroids(all_output, all_sentences)
    # create_embedding_plot(all_output, all_sentences, centroids)
    
    