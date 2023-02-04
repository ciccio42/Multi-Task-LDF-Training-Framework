from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, PlateWithHoleObject, PotWithHandlesObject, HammerObject, BottleObject
from robosuite_env.objects.custom_xml_objects import SpriteCan, CanObject2, CerealObject3, Banana, CerealObject2
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite_env.sampler import BoundarySampler
from robosuite_env.objects.meta_xml_objects import Hoop, Basketball, BasketballRed, BasketballWhite
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, find_elements


class BasketBall(SingleArmEnv):
    """
    This class corresponds to the stacking task for a single robot arm.
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:
            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"
            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param
            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized
        reward_shaping (bool): if True, use dense rewards.
        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
            self,
            robots,
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            initialization_noise="default",
            table_full_size=(1, 1, 0.05),
            table_friction=(1., 5e-3, 1e-4),
            use_camera_obs=True,
            use_object_obs=True,
            reward_scale=1.0,
            reward_shaping=False,
            placement_initializer=None,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            horizon=1000,
            ignore_done=False,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            hoop_id=0,
            ball_id=0,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.hoop_id = hoop_id
        self.ball_id = ball_id

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action):
        """
        Reward function for the task.
        Sparse un-normalized reward:
            - a discrete reward of 2.0 is provided if the red block is stacked on the green block
        Un-normalized components if using reward shaping:
            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube
        The reward is max over the following:
            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking
        The sparse reward only consists of the stacking component.
        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale
        Args:
            action (np array): [NOT USED]
        Returns:
            float: reward value
        """
        reward = float(self._check_success())

        return reward


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6347135597996754,6.407631808935752e-08,1.4049677782065237],
            quat=[0.6298190951347351,0.3214462101459503,0.32144695520401,0.6298189759254456]
        )


        # initialize objects of interest

        self.balls = [Basketball(name='ball1'), BasketballRed(name='ball2'), BasketballWhite(name='ball3')]
        pos_list = ["0.15 0.35 0.08", "-0.07 0.35 0.08", "0.15 -0.35 0.08", "-0.07 -0.35 0.08"]
        name_list = ['place1', 'place2', 'place3', 'place4']
        self.hoop_list = []
        for i in range(4):
            self.hoop_list.append(Hoop(name=name_list[i]))
            hoop_obj = self.hoop_list[i].get_obj()
            hoop_obj.set("pos", pos_list[i])
            if i > 1:
                hoop_obj.set("quat", "0 0 0 1")
            body = find_elements(root=mujoco_arena.worldbody, tags="body", attribs={"name": "table"}, return_first=True)
            body.append(hoop_obj)

        # Create placement initializer

        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.balls,
        )

        for i in range(4):
            self.model.merge_assets(self.hoop_list[i])

        compiler = self.model.root.find('compiler')
        compiler.set('inertiafromgeom', 'auto')
        if compiler.attrib['inertiagrouprange'] == "0 0":
            compiler.attrib.pop('inertiagrouprange')

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            BoundarySampler(
                name="ObjectSampler",
                mujoco_objects=self.balls,
                x_range=[-0.17, 0.12],
                y_range=[-0.12, 0.12],
                rotation=[0, 0 + 1e-4],
                rotation_axis='z',
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.03,
                addtional_dist=0.03
            )
        )


    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

        # Additional object references from this env
        self.target_obj_body_id = self.sim.model.body_name2id(self.balls[self.ball_id].root_body)
        self.place_body_id = self.sim.model.body_name2id(self.hoop_list[self.hoop_id].root_body)
        names = ['place1_goal', 'place2_goal', 'place3_goal', 'place4_goal']
        self.target_loc = self.sim.data.site_xpos[self.sim.model.site_name2id(names[self.hoop_id])]

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[-1], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))


    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        Important keys:
            `'robot-state'`: contains robot-centric information.
            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.
            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.
            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation
        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()
        if self.use_camera_obs:
            cam_name = self.camera_names[0]
            di['image'] = di[cam_name + '_image'].copy()
            del di[cam_name + '_image']
            if self.camera_depths[0]:
                di['depth'] = di[cam_name + '_depth'].copy()
                di['depth'] = ((di['depth'] - 0.95) / 0.05 * 255).astype(np.uint8)

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix
            pr = self.robots[0].robot_model.naming_prefix

            # position and rotation of the first cube
            target_obj_pos = np.array(self.sim.data.body_xpos[self.target_obj_body_id])
            target_obj_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.target_obj_body_id]), to="xyzw"
            )
            di["target_obj_pos"] = target_obj_pos
            di["target_obj_quat"] = target_obj_quat

            # position and rotation of the second cube
            place_pos = np.array(self.sim.data.body_xpos[self.place_body_id])
            place_quat = convert_quat(
                np.array(self.sim.data.body_xquat[self.place_body_id]), to="xyzw"
            )
            di["place_pos"] = place_pos
            di["place_quat"] = place_quat

            # relative positions between gripper and objects
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id])
            di[pr + "gripper_to_target_obj"] = gripper_site_pos - target_obj_pos
            di[pr + "gripper_to_place"] = gripper_site_pos - place_pos
            di["target_obj_to_place"] = target_obj_pos - place_pos

            di["object-state"] = np.concatenate(
                [
                    target_obj_pos,
                    target_obj_quat,
                    place_pos,
                    place_quat,
                    di[pr + "gripper_to_target_obj"],
                    di[pr + "gripper_to_place"],
                    di["target_obj_to_place"],
                ]
            )
        return di

    def _check_success(self):
        """
        Check if blocks are stacked correctly.
        Returns:
            bool: True if blocks are correctly stacked
        """
        target_obj_pos = np.array(self.sim.data.body_xpos[self.target_obj_body_id])

        if np.linalg.norm(target_obj_pos-self.target_loc) < 0.03:
            return True
        else:
            return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.hoop_list[self.hoop_id])

class PandaBasketball(BasketBall):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, ball_id=None, hoop_id=None, **kwargs):
        if ball_id is None:
            ball_id = np.random.randint(0, 3)
        if hoop_id is None:
            hoop_id = np.random.randint(0, 4)
        super().__init__(robots=['Panda'], ball_id=ball_id, hoop_id=hoop_id, **kwargs)

class SawyerBasketball(BasketBall):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, ball_id=None, hoop_id=None, **kwargs):
        if ball_id is None:
            ball_id = np.random.randint(0, 3)
        if hoop_id is None:
            hoop_id = np.random.randint(0, 4)
        super().__init__(robots=['Sawyer'], ball_id=ball_id, hoop_id=hoop_id, **kwargs)

if __name__ == '__main__':
    from robosuite.environments.manipulation.pick_place import PickPlace
    import robosuite
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="IK_POSE")
    env = PandaBasketball(has_renderer=True, controller_configs=controller,
                            has_offscreen_renderer=False,
                            reward_shaping=False, use_camera_obs=False,render_camera='frontview')
    env.reset()
    for i in range(1000):
        if i % 200 == 0:
            env.reset()
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        env.step(action)
        env.render()