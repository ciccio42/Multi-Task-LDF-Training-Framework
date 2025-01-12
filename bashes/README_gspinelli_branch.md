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

<!-- Cond Module -->
## to: 1) generate paths and 2) embeddings from the commands of the finetuning dataset
* run `bashes/generate_train_val_paths_finetuning.sh`
## [train] cond module on the embeddings of the finetuning dataset
* dataset, batchsampler are defined in `Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/datasets/command_encoder/command_encoder_dataset.py`
* for training, run the bash script `train_cond_module_finetuning` in `bashes/` 
# [testing] cond-module via T-SNE plot
* run bash script `test_cond_module_finetuning` -> plots of the embedding (and centroids) are saved in `bashes/finetuning_centroid_figures` folder

<!-- RT1 -->
## create (demo, traj) couples for rt1 training
* run `bashes/generate_traj_demo_couples_from_json.sh`
## [train] RT-1 video-conditioned on the finetuning dataset
* dataset, batchsampler are defined in `training/multi_task_il/datasets/finetuning_paired_dataset.py`
* to launch training, run bash script `train_RT1_video_cond_finetuning.sh` in `bashes/`

# [testing] RT-1
in order to test
* open `test_RT1_video_cond.sh` bash script
* select the root of your project with `PROJECT_NAME` var
* select the step(s)
* run with `nohup ./test_RT1_video_cond.sh > test_RT1_video_cond_module_freezed_output_epoch90.txt &`
* get the results in the same folder where the checkpoints are stored

# create video from test
* navigate to `multi-task-lfd-framework` folder
* select the project folder
* execute `create_video_from_test.sh` script

# visualize finetuning dataset
* run python script `training/multi_task_il/datasets/command_encoder/visualize_actions.py`

# visualization
## visualize video + action for a single dataset
`visualize_action_single_dataset.py`
## visualize video + action for all the datasets
`visualize_actions.py`
## visualize action (no video) for all the datasets
`visualize_actions_static.py`

# datasets
## to convert original real and sim dataset to delta actions
`bashes/convert_real_sim_delta.sh`
## to find min and max for the actions components for every dataset
`bashes/find_min_max_actions.sh`
## to plot histogram for bin distribution given an interval
`bashes/plot_bin_histogram.sh`

### how the couples have been generated
explain
image

## heatmap
heatmap

## come sono fatti batch sampler

