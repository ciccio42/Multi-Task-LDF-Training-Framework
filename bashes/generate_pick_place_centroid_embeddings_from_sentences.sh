#!/bin/bash

# muse and tokenizer
PATH_TO_PT_MODEL="../training/multi_task_il/models/muse/models/model.pt"
PATH_TO_TF_MODEL="../training/multi_task_il/models/muse/models/universal-sentence-encoder-multilingual-large-3"

# in order to save the embeddings in your specified folder, you have to define 2 arguments:
# --save_embedding_path=[YOUR/FOLDER]
# --save_in_a_chosen_folder

python -u ../training/multi_task_il/models/muse/t_sne.py \
        --command_file_path='../training/multi_task_il/models/muse/commands/command_files_extended_11-01_21:01.json' \
        --path_to_tokenizer=${PATH_TO_TF_MODEL} \
        --path_to_muse=${PATH_TO_PT_MODEL} \
        --read_from_json \
        --save_centroids \
        --save_embedding_path='/raid/home/frosa_Loc/opt_dataset/pick_place/new_centroids_commands_embs' \
        --save_in_a_chosen_folder
