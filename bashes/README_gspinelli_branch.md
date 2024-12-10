# info on bash scripts

## to generate embeddings/centroid embeddings from Universal Sentence encoder

* run `generate_pick_place_centroid_embeddings_from_sentences.sh`
* embeddings are saved for every subtask in `/raid/home/frosa_Loc/opt_dataset/pick_place/new_centroids_commands_embs`
* you can run the USE on a json of textual commands, such as `../training/multi_task_il/models/muse/commands/command_files_extended_11-01_21:01.json`
* [TODO] or you can run on the commands of the finetuning dataset

## train condmodule
run `train_cond_module.sh` *bash* script

## test condmodule
run `training/train_scripts/test_cond_module.py` *python* script

## train RT-1

run `train_RT1_video_cond` *bash* script

* for debug, run `train_RT1_video_cond_debug.sh`
* for resume from a checkpoint, run `train_RT1_video_cond_resume.sh`
* [TODO] for training on berkley, run `train_RT1_video_cond_resume.sh`

## test RT-1 with rollouts
run `bashes/test_RT1_video_cond.sh`
* results are saved in `/user/frosa/multi_task_lfd/checkpoint_save_folder`
* *to download videos from traj_--.pkl results*, run `/raid/home/frosa_Loc/Multi-Task-LFD-Framework/utils/analysis/create_video_from_test.sh`



# finetuning

## to generate embeddings from the commands of the finetuning dataset

* run `bashes/generate_train_val_paths_finetuning.sh`

## train cond module on the embeddings of the finetuning dataset

* run

## train RT-1 video-conditioned on the finetuning dataset