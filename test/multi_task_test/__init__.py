# from multi_task_test.eval_functions import *
import os
import json
import multi_task_robosuite_env as mtre
from multi_task_robosuite_env.controllers.controllers.expert_nut_assembly import \
    get_expert_trajectory as nut_expert
from multi_task_robosuite_env.controllers.controllers.expert_pick_place import \
    get_expert_trajectory as place_expert

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
        'env_fn':   place_expert,
        'agent-teacher': ('UR5e_PickPlaceDistractor', 'Panda_PickPlaceDistractor'),
        'render_hw': (200, 360),  # (150, 270)
        'object_set': 2,
    },
    # 'stack_block': {
    #     'num_variations':   6,
    #     'env_fn':   stack_expert,
    #     'eval_fn':  block_stack_eval,
    #     'agent-teacher': ('PandaBlockStacking', 'SawyerBlockStacking'),
    #     'render_hw': (100, 180),  # older models used 100x200!!
    # },
    # 'drawer': {
    #     'num_variations':   8,
    #     'env_fn':   draw_expert,
    #     'eval_fn':  draw_eval,
    #     'agent-teacher': ('PandaDrawer', 'SawyerDrawer'),
    #     'render_hw': (100, 180),
    # },
    # 'button': {
    #     'num_variations':   6,
    #     'env_fn':   press_expert,
    #     'eval_fn':  press_button_eval,
    #     'agent-teacher': ('PandaButton', 'SawyerButton'),
    #     'render_hw': (100, 180),
    # },
    # 'door': {
    #     'num_variations':   4,
    #     'env_fn':   door_expert,
    #     'eval_fn':  open_door_eval,
    #     'agent-teacher': ('PandaDoor', 'SawyerDoor'),
    #     'render_hw': (100, 180),
    # },
    # 'basketball': {
    #     'num_variations':   12,
    #     'env_fn':   basketball_expert,
    #     'eval_fn':  basketball_eval,
    #     'agent-teacher': ('PandaBasketball', 'SawyerBasketball'),
    #     'render_hw': (100, 180),
    # },

}
