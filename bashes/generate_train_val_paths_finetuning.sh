#!/bin/bash


DATASET_FOLDER='/user/frosa/multi_task_lfd/datasets'

# embeddings centroid  for subtasks
GENERATE_CENTROIDS_EMBEDDINGS=true


# muse and tokenizer
PATH_TO_PT_MODEL="../training/multi_task_il/models/muse/models/model.pt"
PATH_TO_TF_MODEL="../training/multi_task_il/models/muse/models/universal-sentence-encoder-multilingual-large-3"

DEBUG=false


# python -u ../training/multi_task_il/datasets/command_encoder/generate_train_val_paths_finetuning.py \
#         --dataset_folder=${DATASET_FOLDER} \
#         --write_train_pkl_path \
#         --write_val_pkl_path \
#         --write_all_pkl_path \
#         --split='0.9,0.1'

if [ $GENERATE_CENTROIDS_EMBEDDINGS == true ]; then 
        python -u ../training/multi_task_il/datasets/command_encoder/query_centroids_embeddings_from_use.py \
        --task_json='./all_pkl_paths.json' \
        --debug=${DEBUG} \
        --path_to_tokenizer=${PATH_TO_TF_MODEL} \
        --path_to_muse=${PATH_TO_PT_MODEL}
fi
