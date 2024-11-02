
n_task = 1
n_subtasks = 16
n_sentence_per_subtask = 30

base_sentence = ''

from random import randint
import json

command_file = {}

tasks = ['pick_place']

for task_id in range(0, len(tasks)):
    command_file[tasks[task_id]] = {}

for task_id in range(0, len(tasks)):
    for subtask_idx in range(0, n_subtasks):
        command_file[tasks[task_id]][str(subtask_idx)] = []
    
green_box_sentences = ['the green box']
yellow_box_sentences = ['the yellow box']
blue_box_sentences = ['the blue box']
red_box_sentences = ['the red box']

first_bin = ['the first bin']
second_bin = ['the second bin']
third_bin = ['the third bin']
fourth_bin = ['the fourth bin']

box_words = {
    0:green_box_sentences,
    1:yellow_box_sentences,
    2:blue_box_sentences,
    3:red_box_sentences
}

bin_words = {
    0:first_bin,
    1:second_bin,
    2:third_bin,
    3:fourth_bin
}


base_sentence = ['place <the_object> into <position>',
                'retrieve <the_object> into <position>',
                'hold <the_object> and transfer it into <position>',
                'Insert <the_object> into <position>',
                'Place <the_object> at <position>',
                'Set <the_object> in <position>',
                'Position <the_object> at <position>',
                'Put <the_object> into <position>',
                'Arrange <the_object> in <position>',
                'Move <the_object> to <position>',
                'Drop <the_object> into <position>']

for _task_id in range(len(tasks)):
    for _subtask_idx in range(n_subtasks): # 0,1,2,3,4,..,15
        _box_id = _subtask_idx // 4 # box del sottotask
        _bin_id = _subtask_idx % 4 # bin del sottotask
        for _sentence_idx in range(n_sentence_per_subtask):
            _sentence = base_sentence[randint(0, len(base_sentence) - 1)]
            _sentence = _sentence.replace('<position>', bin_words[_bin_id][randint(0, len(bin_words[_bin_id]) - 1)])
            _sentence = _sentence.replace('<the_object>', box_words[_box_id][randint(0, len(box_words[_box_id]) - 1)])
            command_file[tasks[_task_id]][str(_subtask_idx)].append(_sentence)

# print(sentence)
import os
_dir = os.getcwd()
_dir = _dir + "/training/multi_task_il/models/muse/commands"
from datetime import datetime
print("saving commands...")
ts = datetime.now().strftime("%m-%d_%H:%M")

with open(f"{_dir}/command_files_extended_{ts}.json", "w") as outfile:
    json.dump(command_file, outfile, indent=4)