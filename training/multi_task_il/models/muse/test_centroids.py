import pickle
import os

FILE_PATH = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/muse/centroids_commands_embs/task_00/centroid_embedding_5.pkl'

for _id in range(16):
    FILE_PATH = FILE_PATH.replace('task_00','task_{:02d}'.format(_id))
    file = open(FILE_PATH, 'rb')
    data = pickle.load(file)
    sentence = data['sentence']
    print("task_{:02d}, sentence: {}".format(_id, sentence))
    FILE_PATH = '/raid/home/frosa_Loc/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training/multi_task_il/models/muse/centroids_commands_embs/task_00/centroid_embedding_0.pkl'
    

