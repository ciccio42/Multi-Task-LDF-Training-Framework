# from multi_task_test.eval_functions import *
import os
import json
import multi_task_robosuite_env as mtre
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert
from multi_task_robosuite_env.controllers.controllers.expert_button import \
    get_expert_trajectory as button_expert
from multi_task_robosuite_env.controllers.controllers.expert_block_stacking import \
    get_expert_trajectory as stack_block_expert

commad_path = os.path.join(os.path.dirname(
    mtre.__file__), "../collect_data/command.json")
with open(commad_path) as f:
    TASK_COMMAND = json.load(f)

ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_names': ['bin_box_1', 'bin_box_2', 'bin_box_3', 'bin_box_4'],
        'ranges': [[-0.255, -0.195], [-0.105, -0.045], [0.045, 0.105], [0.195, 0.255]],
        'splitted_obj_names': ['green box', 'yellow box', 'blue box', 'red box'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15]},
    },
    'nut_assembly': {
        'obj_names': ['round-nut', 'round-nut-2', 'round-nut-3'],
        'peg_names': ['peg1', 'peg2', 'peg3'],
        'splitted_obj_names': ['grey nut', 'brown nut', 'blue nut'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },
    'stack_block': {
        'obj_names': ['cubeA', 'cubeB', 'cubeC'],
    },
    'button': {
        'obj_names': ['machine1_goal1', 'machine1_goal2', 'machine1_goal3',
                      'machine2_goal1', 'machine2_goal2', 'machine2_goal3'],
        'place_names': ['machine1_goal1_final', 'machine1_goal2_final', 'machine1_goal3_final',
                        'machine2_goal1_final', 'machine2_goal2_final', 'machine2_goal3_final']
    }
}


TASK_MAP = {
    'nut_assembly':  {
        'num_variations':   9,
        'env_fn':   nut_expert,
        'agent-teacher': ('UR5e_NutAssemblyDistractor', 'Panda_NutAssemblyDistractor'),
        'render_hw': (200, 360),
        'object_set': 1,
    },
    'pick_place': {
        'num_variations':   16,
        'num_variations_per_object':   4,
        'env_fn':   place_expert,
        'agent-teacher': ('UR5e_PickPlaceDistractor', 'Panda_PickPlaceDistractor'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 2,
    },
    'stack_block': {
        'num_variations':   6,
        'env_fn':   stack_block_expert,
        'agent-teacher': ('UR5e_BlockStacking', 'Panda_BlockStacking'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 1,
    },
    'button': {
        'num_variations':   6,
        'env_fn':   button_expert,
        'agent-teacher': ('UR5e_Button', 'Panda_Button'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 1,
    },
}
