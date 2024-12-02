from multi_task_il.datasets.command_encoder.multi_task_command_encoder import CommandEncoderDataset, CommandEncoderSampler
from multi_task_il.datasets.command_encoder.cond_module import CondModule
from torch.utils.data import DataLoader

import torch
from torch.nn import CosineEmbeddingLoss
from torch.optim import SGD
from torch.optim import AdamW, RMSprop
from torch.optim import lr_scheduler

import wandb
import random
import numpy as np
import cv2

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

    
def compute_cosine_similarity(output_embedding, gt_embedding, batch_size, target):
        # capita che l'ultimo batch non ha la stessa dimensione del batch_size
    if output_embedding.shape[0] == gt_embedding.shape[0] and batch_size != output_embedding.shape[0]:
        # print(f"changing target tensor to size {output_embedding.shape[0]}")
        target = torch.ones(output_embedding.shape[0]).to(device)
        loss = cosine_loss(output_embedding, gt_embedding, target)
        target = torch.ones(batch_size).to(device)
    else:
        # loss = cosine_loss(output_embedding, gt_embedding, target)
        loss = cosine_loss(output_embedding, gt_embedding, target)
        
    return loss, target

def get_train_val_loader(data_augs, batch_size, robot='panda',shuffle=False):
    train_dataset = CommandEncoderDataset(data_augs=data_augs,
                                          robot=robot,
                                          mode='train',
                                          use_embedding_centroids=True,
                                          n_train_and_val_samples_per_subtask=60)
    val_dataset = CommandEncoderDataset(data_augs=data_augs,
                                        robot=robot,
                                        mode='val',
                                        use_embedding_centroids=True,
                                        n_train_and_val_samples_per_subtask=60)
    # train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    train_sampler = CommandEncoderSampler(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_sampler = CommandEncoderSampler(val_dataset, batch_size=batch_size, shuffle=shuffle)
    # with sampler
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    # no sampler
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    return train_loader, val_loader

def get_cond_module():
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 256, 512]).to(device)
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 256, 512], pretrained=True).to(device) # prossimo da provare
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512], pretrained=True).to(device)
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 256, 128, 256, 512], pretrained=True).to(device)
    cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512, 512, 512], pretrained=True).to(device)
    ######## provare anche resNet
    # cond_module = CondModule(model_name='r2plus1d_18', demo_linear_dim=[512]).to(device) # non sotto lo 0.35 val
    
    ##### geliamo la backbone se il modello è pre-trainato
    for name, par in cond_module._backbone.named_parameters():
        par.requires_grad = True
        
    return cond_module

def get_optimizer_lr_scheduler():
        #### lr scheduler
    if LR_SCHEDULER:
        # optimizer1 = torch.optim.Adam(cond_module.parameters(), lr=0.1)
        #####################################################################
        optimizer1 = AdamW(params=cond_module.parameters(), lr=STARTING_LR_OPTIMIZER_1) ####### 
        #####################################################################
        if TWO_OPTIMIZERS:
            optimizer2 = torch.optim.SGD(cond_module.parameters(), lr=0.1, momentum=MOMENTUM)
        # optimizer2 = torch.optim.SGD(cond_module.parameters(), lr=0.01)
        else:
            optimizer2 = None
        
        if TWO_OPTIMIZERS:
            scheduler = lr_scheduler.MultiStepLR(optimizer1,
                                            milestones=[5,20], #0.01, 0.001
                                            gamma=0.1)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer1,
                                            # milestones=[20,40,80], #0.05, 0.025, 0.00125
                                            # milestones=[50,150,300], #0.05, 0.025, 0.00125 EPOCHS = 500
                                            # milestones=[100,250,500], #0->0.1, 100->0.05, 250->0.025, 500->0.00125 EPOCHS = 1000
                                            # milestones=[50,100,250], #0->0.1, 50->0.05, 100->0.025, 250->0.00125 EPOCHS = 1000
                                            # milestones=[50,250,500], #start lr 0->0.1, 50->0.05, 250->0.025, 500->0.00125 EPOCHS = 1000
                                            # milestones=[50,250,500], #start lr 0->0.01, 50->0.005, 250->0.0025, 500->0.000125 EPOCHS = 1000
                                            # milestones=[10,50,250], #start lr 0->0.1, 10>0.01, 50->0.001, 250->0.0001
                                            # milestones=[100,250,500], #start lr 0->0.1, 100>0.05, 250->0.025 500->0.0125
                                            milestones=[100,250], #start lr 0->0.01 100->0.005 250->0.0025
                                            gamma=0.5)
            
            ###############################################
            # PROVARE EXPONENTIAL LR
            ###############################################
            
        
       # optimizer = optimizer1
        
        return scheduler, optimizer1, optimizer2
    else:
        # optimizer = SGD(cond_module.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        optimizer = AdamW(params=cond_module.parameters(), lr=LEARNING_RATE)
        # optimizer = RMSprop(params=cond_module.parameters(), lr=LEARNING_RATE)
        
        return None, optimizer, None
    
def create_save_path():
    # scusate
    import os
    root_dir = os.getcwd()
    assert root_dir == '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework', \
        'start script from Multi-Task-LFD-Training-Framework dir'
    save_path = 'training/train_scripts/command_encoder/models'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    subfolders = [f'{save_path}/{ROBOT}',
                  f'/{str(BATCH_SIZE)}_batch_size',
                  f'/{str(EPOCHS)}_epochs',
                  f'/{str(STARTING_LR_OPTIMIZER_1)}_lr']
    
    for _idx_subf, name_subf in enumerate(subfolders):
        
        if _idx_subf == 0:
            curr_dir = name_subf
            if not os.path.isdir(curr_dir):
                os.mkdir(curr_dir)
        else:
            curr_dir += name_subf
            if not os.path.isdir(curr_dir):
                os.mkdir(curr_dir)
            
    save_path = curr_dir

    # curr_dir += 
    # if not os.path.isdir(curr_dir):
    #     os.mkdir(curr_dir)
    # curr_dir += 
    # if not os.path.isdir(curr_dir):
    #     os.mkdir(curr_dir)
    # curr_dir += 
    # if not os.path.isdir(curr_dir):
    #     os.mkdir(curr_dir)
    # curr_dir += 
    # if not os.path.isdir(curr_dir):
    #     os.mkdir(curr_dir)
    # save_path = curr_dir
    
    assert os.path.isdir(save_path) == True, f"{save_path} is not a valid save dir"
    print("save folder for the model: {}".format(save_path))
    
    return save_path


if __name__ == '__main__':
    
    
    DEBUG = False
    CUDA_DEVICE = 1
    USE_CENTROIDS = True
    # training parameters
    # EPOCHS = 150
    EPOCHS = 5
    # BATCH_SIZE = 32
    # BATCH_SIZE = 16
    # BATCH_SIZE = 16
    # BATCH_SIZE = 32
    # BATCH_SIZE = 4
    BATCH_SIZE = 32
    SHUFFLE = True
    # LEARNING_RATE = 0.001
    # LEARNING_RATE = 0.001
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9 # for SGD
    USE_WANDB = True
    EARLY_STOP = 2
    
    LR_SCHEDULER = True
    ### two optimizers
    TWO_OPTIMIZERS = False
    EPOCH_CHANGE_OPTIMIZER = 29
    
    # STARTING_LR_OPTIMIZER_1 = 0.01
    # STARTING_LR_OPTIMIZER_1 = 0.001
    # STARTING_LR_OPTIMIZER_1 = 0.00001
    STARTING_LR_OPTIMIZER_1 = 0.0001
    
    # ROBOT='panda'
    ROBOT='panda'
    
    LR_1 = 0.001
    LR_2 = 0.01
    
    #### cuda device
    print(f"current device: {torch.cuda.current_device()}")
    if CUDA_DEVICE != 0:
        print(f"switching to {CUDA_DEVICE}...")
        torch.cuda.set_device(CUDA_DEVICE)
        print(f"current device: {torch.cuda.current_device()}")
    
    if USE_WANDB:
        wandb.login()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, val_loader = get_train_val_loader(DATA_AUGS, BATCH_SIZE, robot=ROBOT, shuffle=True) # we set shuffle=True because we want to shuffle batch indexes with sampler
    cond_module = get_cond_module()
    
    # cosine_loss = CosineEmbeddingLoss(reduction='none')
    cosine_loss = CosineEmbeddingLoss()
    target = torch.ones(BATCH_SIZE).to(device) # se target 1, l'obietto è massimizzare la cosine similarity
        
    scheduler, optimizer, optimizer2 = get_optimizer_lr_scheduler()
    save_path = create_save_path()
    best_vloss = np.inf
    
    
    if USE_WANDB:
        run = wandb.init(
        # Set the project where this run will be logged
        project="cond_module",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "momentum": MOMENTUM
        },
        )
        
    training_steps_per_epoch = len(train_loader) / BATCH_SIZE
    
    best_val_counter = 0
    
    for epoch in range(EPOCHS):
        
        if LR_SCHEDULER and epoch == EPOCH_CHANGE_OPTIMIZER and TWO_OPTIMIZERS:
            scheduler = lr_scheduler.MultiStepLR(optimizer2,
                                            milestones=[5, 30], #0.01, 0.001
                                            gamma=0.1)
            
            optimizer = optimizer2
        
        if LR_SCHEDULER:
            print(f"epoch {epoch + 1}, current lr {scheduler.get_last_lr()}")
        else:
            print(f'epoch {epoch + 1}')
        
        ####### train loop
        cond_module.train()
        running_loss = 0.
        last_loss = 0.
        for _i, data in enumerate(train_loader):
            
            if not USE_CENTROIDS:
                video_input, gt_embedding = data['demo_data']['demo'], data['embedding']['embedding']
            else:
                # video_input, gt_embedding = data['demo_data']['demo'], data['embedding']
                video_input, gt_embedding = data['demo_data']['demo'], data['embedding']['centroid_embedding']
            video_input, gt_embedding = video_input.to(device), gt_embedding.to(device).squeeze(1)
            optimizer.zero_grad()
            output_embedding = cond_module(video_input)
            if DEBUG:
                if _i == 0:
                    for indx, sample in enumerate(video_input):
                        # i-th sample
                        for t in range(4):
                            if t == 3:
                                img_tensor = np.moveaxis(sample[t].detach().cpu().numpy()*255, 0, -1)
                                print(data['embedding']['sentence'])
                                cv2.imwrite(f"{indx}_frame_{t}.png", img_tensor)
            

            
            
            loss, target = compute_cosine_similarity(output_embedding, gt_embedding, BATCH_SIZE, target)
            loss.backward()
            
            optimizer.step()

            running_loss += loss.item()
            
        train_steps_this_epoch = _i
                
        ####### validation loop
        cond_module.eval()
        avg_val_loss = 0.
        
        with torch.no_grad():
            for _i, data in enumerate(val_loader):
                if not USE_CENTROIDS:
                    video_input, gt_embedding = data['demo_data']['demo'], data['embedding']['embedding']
                else:
                    # video_input, gt_embedding = data['demo_data']['demo'], data['embedding']
                    video_input, gt_embedding = data['demo_data']['demo'], data['embedding']['centroid_embedding']
                video_input, gt_embedding = video_input.to(device), gt_embedding.to(device).squeeze(1)
                output_embedding = cond_module(video_input)
                ############################# fai un check di target
                val_loss, target = compute_cosine_similarity(output_embedding, gt_embedding, BATCH_SIZE, target)
                avg_val_loss+=val_loss
                #avg_accuracy
                
        avg_val_loss = avg_val_loss / (_i + 1)
        #avg_accuracy

        # compute training loss for the entire epoch
        assert (training_steps_per_epoch - 1) == train_steps_this_epoch, "something wrong"
        if train_steps_this_epoch == (training_steps_per_epoch - 1):
            last_loss = running_loss / training_steps_per_epoch
            print('train loss: {}'.format(last_loss))
            running_loss = 0.
            if USE_WANDB:
                wandb.log({'train_loss' : last_loss, 'val_loss' : avg_val_loss})
        
        best_val_counter+=1 # conta il numero di volte che la val_loss non migliora a ogni epoca
        
        if avg_val_loss < best_vloss:
            print("best new val loss: {}".format(avg_val_loss))
            best_vloss = avg_val_loss
            best_val_counter = 0
            
        if best_val_counter == EARLY_STOP:
            print(f"model has not been learning for {EARLY_STOP} epochs, breaking train_val loop...")
            break # exit from train_val loop
            
        ## step lr_scheduler at the end of the epoch
        if LR_SCHEDULER:
            scheduler.step()    
            
    from datetime import datetime
    print("saving model...")
    ts = datetime.now().strftime("%m-%d_%H:%M")
    # save model
    save_model_path = '{}/cond_module_{}.pth'.format(save_path, ts)
    torch.save(cond_module.state_dict(), save_model_path)
    print("saved model at {}".format(save_model_path))
    
    # farlo più sistemato (un dataframe con la loss ad ogni epoca)
    results_file = open('{}/results.txt'.format(save_path), 'w')
    results_file.write(str(last_loss))
    results_file.close()
    
    # cond_model_loaded = torch.load('{}/cond_module_{}.pth'.format(save_path, ts), weights_only=False)

    # for i in train_loader:
    #     print(i)
    #     video_input = i['demo_data']['demo']
    #     gt_embedding = i['embedding']['embedding']
        
    #     output = cond_module(video_input) # self._backbone è None
    #     gt_embedding = gt_embedding.squeeze(dim=0)
        
    #     loss = cosine_loss(output, gt_embedding, target)
    #     exit()
    
    # train_step = int(EPOCHS *
    #                 int(len(dataset)/BATCH_SIZE))



            # for indx, sample in enumerate(video_input):
            #     # i-th sample
            #     for t in range(4):
            #        if t == 3:
            #         img_tensor = np.moveaxis(sample[t].detach().cpu().numpy()*255, 0, -1)
            #         # print(data['embedding']['sentence'])
            #         # cv2.imwrite(f"{indx}_frame_{t}.png", img_tensor)
            #         if indx == 0:
            #             assert "green" in data['embedding']['sentence'][indx] and 'first' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 1:
            #             assert "green" in data['embedding']['sentence'][indx] and 'second' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 2:
            #             assert "green" in data['embedding']['sentence'][indx] and 'third' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 3:
            #             assert "green" in data['embedding']['sentence'][indx] and 'fourth' in data['embedding']['sentence'][indx], "not order"
                    
            #         if indx == 4:
            #             assert "yellow" in data['embedding']['sentence'][indx] and 'first' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 5:
            #             assert "yellow" in data['embedding']['sentence'][indx] and 'second' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 6:
            #             assert "yellow" in data['embedding']['sentence'][indx] and 'third' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 7:
            #             assert "yellow" in data['embedding']['sentence'][indx] and 'fourth' in data['embedding']['sentence'][indx], "not order"
                    
            #         if indx == 8:
            #             assert "blue" in data['embedding']['sentence'][indx] and 'first' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 9:
            #             assert "blue" in data['embedding']['sentence'][indx] and 'second' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 10:
            #             assert "blue" in data['embedding']['sentence'][indx] and 'third' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 11:
            #             assert "blue" in data['embedding']['sentence'][indx] and 'fourth' in data['embedding']['sentence'][indx], "not order"
                        
            #         if indx == 12:
            #             assert "red" in data['embedding']['sentence'][indx] and 'first' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 13:
            #             assert "red" in data['embedding']['sentence'][indx] and 'second' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 14:
            #             assert "red" in data['embedding']['sentence'][indx] and 'third' in data['embedding']['sentence'][indx], "not order"
            #         if indx == 15:
            #             assert "red" in data['embedding']['sentence'][indx] and 'fourth' in data['embedding']['sentence'][indx], "not order"
                        