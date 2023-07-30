from multi_task_test import startup_env, get_action
import numpy as np


def block_stack_eval(model, env, context, gpu_id, variation_id, img_formatter, max_T=85, baseline=False):
    done, states, images, context, obs, traj, tasks = \
        startup_env(model, env, context, gpu_id,
                    variation_id, baseline=baseline)
    n_steps = 0
    obj_loc = env.sim.data.body_xpos[env.cubeA_body_id]
    obj_key = 'cubeA_pos'
    start_z = obs[obj_key][2]
    while not done:
        tasks['reached'] = tasks['reached'] or np.linalg.norm(
            obj_loc - obs['eef_pos']) < 0.045
        tasks['picked'] = tasks['picked'] or (
            tasks['reached'] and obs[obj_key][2] - start_z > 0.05)
        if baseline and len(states) >= 5:
            states, images = [], []
        states.append(np.concatenate(
            (obs['ee_aa'], obs['gripper_qpos'])).astype(np.float32)[None])
        images.append(img_formatter(obs['camera_front_image'])[None])

        action = get_action(model, states, images, context,
                            gpu_id, n_steps, max_T, baseline)

        obs, reward, env_done, info = env.step(action)
        traj.append(obs, reward, done, info, action)

        tasks['success'] = reward or tasks['success']
        n_steps += 1
        if env_done or reward or n_steps > max_T:
            done = True
    env.close()
    del env
    del states
    del images
    del model

    return traj, tasks
