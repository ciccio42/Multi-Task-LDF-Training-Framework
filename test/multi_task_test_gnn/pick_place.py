import torch
from multi_task_test_gnn.utils import startup_env, create_env_graph
from multi_task_test_gnn.primitive import perform_pick_place_primitive
from collections import OrderedDict
from train_scripts.gnn.utils import compute_accuracy
from multi_task_il_gnn.datasets.utils import NUM_OBJ_NUM_TARGET_PER_OBJ
from colorama import Style
from colorama import Fore
from colorama import init as colorama_init
colorama_init()
OBJECT_SET = 2


def pick_place_eval_gnn(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, controller=None, task_name=None, config=None, perform_augs=True, real=True):

    model_inputs = OrderedDict()

    start_up_env_return = \
        startup_env(model=model,
                    env=env,
                    context=context,
                    gpu_id=gpu_id,
                    variation_id=variation_id,
                    baseline=baseline,
                    )

    done, states, images, context, obs, traj, tasks, gt_obs, current_gripper_pose = start_up_env_return

    graph = create_env_graph(env=env,
                             task_name=task_name,
                             variation_id=variation_id,
                             obs=obs)

    model_inputs['demo'] = context
    model_inputs['node_features'] = graph.x.to(gpu_id)[None]
    # take node class from graph
    class_labels = graph.y
    num_objs = NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][0]
    num_targets = NUM_OBJ_NUM_TARGET_PER_OBJ[task_name][1]

    obj_class = torch.zeros(num_objs+num_targets).to(gpu_id)
    target_class = torch.zeros(num_objs+num_targets).to(gpu_id)
    obj_class[:num_objs] = class_labels[:num_objs]
    target_class[num_objs:] = class_labels[num_objs:]
    # run inference
    out = model(
        inputs=model_inputs,
        inference=True
    )

    # get predicted logits
    if config.lcgnet_conf.BUILD_NODE_CLASSIFIER:
        obj_logits = out[0].squeeze()
        target_logits = out[1].squeeze()

        # compute global accuracy
        obj_accuracy, target_accuracy = compute_accuracy(obj_logits=obj_logits[None],
                                                         target_logits=target_logits[None],
                                                         obj_gt=obj_class[None],
                                                         target_gt=target_class[None])
        tasks['obj_selection'] = obj_accuracy
        tasks['target_selection'] = target_accuracy

        print(f"{Fore.GREEN}Target correctly identified? {obj_accuracy}\nPlacing correctly identified? {target_accuracy}{Style.RESET_ALL}")

        target_indx = obj_logits.argmax()
        placing_indx = target_logits.argmax()
        # take target and placing location position
        target_pos = model_inputs['node_features'][target_indx][:6].cpu(
        ).numpy()
        placing_pos = model_inputs['node_features'][placing_indx][:6].cpu(
        ).numpy()

        print(
            f"{Fore.YELLOW}Running pick-place primitive{target_accuracy}{Style.RESET_ALL}")
        state, success = perform_pick_place_primitive(env=env,
                                                      picking_loc=target_pos,
                                                      placing_loc=placing_pos,
                                                      trajectory=traj)

        return traj, tasks


def pick_place_eval(model, env, gt_env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False, model_name=None, task_name="pick_place", config=None, real=True):

    if env != None:
        from multi_task_robosuite_env.controllers.controllers.expert_pick_place import PickPlaceController
        controller = PickPlaceController(
            env=env.env,
            tries=[],
            ranges=[],
            object_set=OBJECT_SET)
    else:
        controller = None
    pick_place_eval_gnn(model=model,
                        env=env,
                        context=context,
                        gpu_id=gpu_id,
                        variation_id=variation_id,
                        img_formatter=img_formatter,
                        max_T=max_T,
                        baseline=baseline,
                        controller=controller,
                        task_name=task_name,
                        config=config,
                        real=real
                        )
