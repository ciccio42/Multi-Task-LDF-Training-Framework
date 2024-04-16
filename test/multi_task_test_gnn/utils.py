from multi_task_il_gnn.datasets.utils import NUM_FEATURES, NUM_OBJ_NUM_TARGET_PER_OBJ, OBJECTS_POS_DIM, compute_object_features
import os
import random
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms
import cv2
from multi_task_test_gnn import TASK_MAP
from robosuite import load_controller_config
from collections import deque
import robosuite.utils.transform_utils as T
from multi_task_il_gnn.datasets.savers import Trajectory
from robosuite.utils.transform_utils import quat2axisangle
from torch_geometric.data import Data
from colorama import Style
from colorama import Fore
from colorama import init as colorama_init
colorama_init()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_tvf_formatter(config, env_name):
    """Use this for torchvision.transforms in multi-task dataset,
    note eval_fn always feeds in traj['obs']['images'], i.e. shape (h,w,3)
    """

    def resize_crop(img, bb=None, agent=False):
        """applies to every timestep's RGB obs['camera_front_image']"""
        task_spec = config.tasks_cfgs.get(env_name, dict())
        img_height, img_width = img.shape[:2]
        """applies to every timestep's RGB obs['camera_front_image']"""
        if len(getattr(task_spec, "demo_crop", OrderedDict())) != 0 and not agent:
            crop_params = task_spec.get(
                "demo_crop", [0, 0, 0, 0])
        if len(getattr(task_spec, "agent_crop", OrderedDict())) != 0 and agent:
            crop_params = task_spec.get(
                "agent_crop", [0, 0, 0, 0])
        if len(getattr(task_spec, "task_crops", OrderedDict())) != 0:
            crop_params = task_spec.get(
                "task_crops", [0, 0, 0, 0])
        if len(getattr(task_spec, "crop", OrderedDict())) != 0:
            crop_params = task_spec.get(
                "crop", [0, 0, 0, 0])

        top, left = crop_params[0], crop_params[2]
        img_height, img_width = img.shape[0], img.shape[1]
        box_h, box_w = img_height - top - \
            crop_params[1], img_width - left - crop_params[3]

        img = transforms.ToTensor()(img.copy())
        # ---- Resized crop ----#
        img = transforms.functional.resized_crop(img, top=top, left=left, height=box_h,
                                                 width=box_w, size=(config.dataset_cfg.height, config.dataset_cfg.width))

        cv2.imwrite("resized_target_obj.png", np.moveaxis(
            img.numpy()*255, 0, -1))

        return img

    return resize_crop


def select_random_frames(frames, n_select, sample_sides=True, random_frames=True):
    selected_frames = []
    def clip(x): return int(max(0, min(x, len(frames) - 1)))
    per_bracket = max(len(frames) / n_select, 1)

    if random_frames:
        for i in range(n_select):
            n = clip(np.random.randint(
                int(i * per_bracket), int((i + 1) * per_bracket)))
            if sample_sides and i == n_select - 1:
                n = len(frames) - 1
            elif sample_sides and i == 0:
                n = 1
            selected_frames.append(n)
    else:
        for i in range(n_select):
            # get first frame
            if i == 0:
                n = 1
            # get the last frame
            elif i == n_select - 1:
                n = len(frames) - 1
            elif i == 1:
                obj_in_hand = 0
                # get the first frame with obj_in_hand and the gripper is closed
                for t in range(1, len(frames)):
                    state = frames.get(t)['info']['status']
                    trj_t = frames.get(t)
                    gripper_act = trj_t['action'][-1]
                    if state == 'obj_in_hand' and gripper_act == 1:
                        obj_in_hand = t
                        n = t
                        break
            elif i == 2:
                # get the middle moving frame
                start_moving = 0
                end_moving = 0
                for t in range(obj_in_hand, len(frames)):
                    state = frames.get(t)['info']['status']
                    if state == 'moving' and start_moving == 0:
                        start_moving = t
                    elif state != 'moving' and start_moving != 0 and end_moving == 0:
                        end_moving = t
                        break
                n = start_moving + int((end_moving-start_moving)/2)
            selected_frames.append(n)

    return [frames[i]['obs']['camera_front_image'] for i in selected_frames]


def build_env_context(img_formatter, T_context=4, ctr=0, env_name='nut', heights=100, widths=200, size=False, shape=False, color=False, gpu_id=0, variation=None, random_frames=True, controller_path=None, ret_gt_env=False, seed=42):

    print(f"Seed: {seed}")
    if controller_path == None:
        controller = load_controller_config(default_controller='IK_POSE')
    else:
        # load custom controller
        controller = load_controller_config(
            custom_fpath=controller_path)
    # assert gpu_id != -1
    build_task = TASK_MAP.get(env_name, None)
    assert build_task, 'Got unsupported task '+env_name
    div = int(build_task['num_variations'])
    env_fn = build_task['env_fn']
    agent_name, teacher_name = build_task['agent-teacher']

    if variation == None:
        variation = ctr % div
    else:
        variation = variation

    teacher_expert_rollout = env_fn(teacher_name,
                                    controller_type=controller,
                                    task=variation,
                                    seed=seed,
                                    gpu_id=gpu_id,
                                    object_set=TASK_MAP[env_name]['object_set'])

    agent_env = env_fn(agent_name,
                       controller_type=controller,
                       task=variation,
                       ret_env=True,
                       seed=seed,
                       gpu_id=gpu_id,
                       object_set=TASK_MAP[env_name]['object_set'])

    if ret_gt_env:
        gt_env = env_fn(agent_name,
                        controller_type=controller,
                        task=variation,
                        ret_env=True,
                        seed=seed,
                        gpu_id=gpu_id,
                        object_set=TASK_MAP[env_name]['object_set'])

    context = select_random_frames(
        teacher_expert_rollout, T_context, sample_sides=True, random_frames=random_frames)

    context = [img_formatter(i[:, :, ::-1])[None] for i in context]
    # assert len(context ) == 6
    if isinstance(context[0], np.ndarray):
        context = torch.from_numpy(np.concatenate(context, 0))[None]
    else:
        context = torch.cat(context, dim=0)[None]

    if ret_gt_env:
        return agent_env, context, variation, teacher_expert_rollout, gt_env
    else:
        return agent_env, context, variation, teacher_expert_rollout


def get_eval_fn(env_name):
    if "pick_place" in env_name:
        from multi_task_test_gnn.pick_place import pick_place_eval
        return pick_place_eval
    elif "nut_assembly" in env_name:
        from multi_task_test_gnn.nut_assembly import nut_assembly_eval
        return nut_assembly_eval
    elif "button" in env_name:
        from multi_task_test_gnn.button_press import press_button_eval
        return press_button_eval
    elif "stack" in env_name:
        from multi_task_test_gnn.block_stack import block_stack_eval
        return block_stack_eval
    else:
        assert NotImplementedError


def startup_env(model, env, context, gpu_id, variation_id, baseline=None):

    done, states, images = False, [], []
    if baseline is None:
        states = deque(states, maxlen=1)
        images = deque(images, maxlen=1)  # NOTE: always use only one frame
    context = context.cuda(gpu_id).float()

    while True:
        try:
            obs = env.reset()
            # make a "null step" to stabilize all objects
            current_gripper_position = env.sim.data.site_xpos[env.robots[0].eef_site_id]
            current_gripper_orientation = T.quat2axisangle(T.mat2quat(np.reshape(
                env.sim.data.site_xmat[env.robots[0].eef_site_id], (3, 3))))
            current_gripper_pose = np.concatenate(
                (current_gripper_position, current_gripper_orientation, np.array([-1])), axis=-1)
            obs, reward, env_done, info = env.step(current_gripper_pose)
            break
        except:
            pass

    gt_obs = None

    traj = Trajectory()
    traj.append(obs)
    tasks = {'success': False, 'reached': False,
             'picked': False, 'variation_id': variation_id}
    return done, states, images, context, obs, traj, tasks, gt_obs, current_gripper_pose


def create_env_graph(env, task_name, variation_id, obs):
    print(f"{Fore.GREEN} Generating graph for task {task_name}{Style.RESET_ALL}")

    num_entities = NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0] + \
        NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]

    feature_vector = np.zeros((num_entities, NUM_FEATURES))

    for obj_indx, object_name in enumerate(OBJECTS_POS_DIM[task_name]['obj_names']):
        if 'bin' != object_name:
            object_position = obs[f'{object_name}_pos']
            object_orientation = quat2axisangle(obs[f'{object_name}_quat'])
        else:
            object_position = OBJECTS_POS_DIM[task_name]['bin_position']

        if task_name == "pick_place":
            if "bin" not in object_name:
                feature_vector[obj_indx][:3] = object_position
                feature_vector[obj_indx][3:6] = object_orientation
                feature_vector[obj_indx][6:9] = compute_object_features(task_name=task_name,
                                                                        object_name=object_name)
                feature_vector[obj_indx][9] = np.array(
                    obj_indx, dtype=np.uint8)
                feature_vector[obj_indx][10] = np.array(1, dtype=np.uint8)
                feature_vector[obj_indx][11] = 1 if obj_indx == obs['target-object'] else 0

            else:
                # we are considering bins
                bin_pos = OBJECTS_POS_DIM[task_name]['bin_position']
                bins_pos = list()
                bins_pos.append([bin_pos[0],
                                 bin_pos[1]-0.15-0.15/2,
                                 bin_pos[2]])
                bins_pos.append([bin_pos[0],
                                bin_pos[1]-0.15/2,
                                bin_pos[2]])
                bins_pos.append([bin_pos[0],
                                bin_pos[1]+0.15/2,
                                bin_pos[2]])
                bins_pos.append([bin_pos[0],
                                bin_pos[1]+0.15+0.15/2,
                                bin_pos[2]])
                for bin_indx, pos in enumerate(bins_pos):
                    bin_indx_relative = bin_indx + \
                        (len(OBJECTS_POS_DIM[task_name]['obj_names'])-1)
                    feature_vector[bin_indx_relative][:3] = pos
                    feature_vector[bin_indx_relative][3:
                                                      6] = np.array([.0, .0, .0])
                    feature_vector[bin_indx_relative][6:9] = compute_object_features(task_name=task_name,
                                                                                     object_name=object_name)
                    feature_vector[bin_indx_relative][9] = np.array(
                        bin_indx+bin_indx_relative, dtype=np.uint8)
                    feature_vector[bin_indx_relative][10] = np.array(
                        0, dtype=np.uint8)
                    feature_vector[bin_indx_relative][11] = 1 if bin_indx == obs['target-box-id'] else 0

    # 2. Fill the edge index vector
    # connect each object with a target position
    # edge_indx has shape 2*num_edge
    num_edges = (NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0] *
                 NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]) * 2
    edge_indx = np.zeros((2, num_edges))

    num_edge = 0
    for object_indx in range(NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]):
        for place_indx in range(NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]):
            # undirected graph edges
            edge_indx[0][num_edge] = object_indx
            edge_indx[1][num_edge] = place_indx + \
                NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]  # offset in node indices
            num_edge += 1

            edge_indx[0][num_edge] = place_indx + \
                NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]
            edge_indx[1][num_edge] = object_indx
            num_edge += 1

    graph = Data(x=torch.tensor(feature_vector[:, :-1]),
                 edge_index=torch.tensor(edge_indx),
                 y=torch.tensor(feature_vector[:, -1]))

    return graph
