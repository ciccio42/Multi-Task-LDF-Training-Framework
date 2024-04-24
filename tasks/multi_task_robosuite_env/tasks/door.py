from collections import OrderedDict
import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.arenas import TableArena
multi_task_robosuite_env.objects.custom_xml_objects import DoorObject, DoorObject2, SmallDoorObject, SmallDoorObject2
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler



class DoorEnv(SingleArmEnv):
    """
    This class corresponds to the door opening task for a single robot arm.
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
        use_latch (bool): if True, uses a spring-loaded handle and latch to "lock" the door closed initially
            Otherwise, door is instantiated with a fixed handle
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
            use_latch=False,
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
            task_id=0,
    ):
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.table_full_size = (0.9, 1, 0.05)
        self.table_offset = (0, 0, 0.8)

        # reward configuration
        self.use_latch = use_latch
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.task_id = task_id

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

    def reward(self, action=None):
        """
        Reward function for the task.
        Sparse un-normalized reward:
            - a discrete reward of 1.0 is provided if the door is opened
        Un-normalized summed components if using reward shaping:
            - Reaching: in [0, 0.25], proportional to the distance between door handle and robot arm
            - Rotating: in [0, 0.25], proportional to angle rotated by door handled
              - Note that this component is only relevant if the environment is using the locked door version
        Note that a successfully completed task (door opened) will return 1.0 irregardless of whether the environment
        is using sparse or shaped rewards
        Note that the final reward is normalized and scaled by reward_scale / 1.0 as
        well so that the max score is equal to reward_scale
        Args:
            action (np.array): [NOT USED]
        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

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
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        # initialize objects of interest
        self.doors = [SmallDoorObject(
            name="door1",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
        ), SmallDoorObject2(
            name="door2",
            friction=0.0,
            damping=0.1,
            lock=self.use_latch,
        )
        ]

        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.doors,
        )

        compiler = self.model.root.find('compiler')
        compiler.set('inertiafromgeom', 'auto')
        if compiler.attrib['inertiagrouprange'] == "0 0":
            compiler.attrib.pop('inertiagrouprange')

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds.
        """
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # each object should just be sampled in the bounds of the bin (with some tolerance
        x_range = [[0.0, 0.04], [-0.05, -0.01]]
        y_range = [[-0.30, -0.26], [0.27, 0.32]]
        rotation = [(-np.pi / 2. - 0.005, -np.pi / 2.+0.005), (np.pi / 2.-0.005, np.pi / 2. + 0.005)]
        for i in range(2):
            self.placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name=f"{i}Sampler",
                    mujoco_objects=self.doors[i],
                    x_range=x_range[i],
                    y_range=y_range[i],
                    rotation=rotation[i],
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=self.table_offset,
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
        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.doors[self.task_id // 2].door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.doors[self.task_id // 2].frame_body)
        self.object_body_ids["latch"] = self.sim.model.body_name2id(self.doors[self.task_id // 2].latch_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.doors[self.task_id // 2].important_sites["handle"])
        self.door_handle_start_id = self.sim.model.site_name2id(self.doors[self.task_id // 2].important_sites["handle_start"])
        self.door_handle_end_id = self.sim.model.site_name2id(self.doors[self.task_id // 2].important_sites['handle_end'])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.doors[self.task_id // 2].joints[0])
        if self.use_latch:
            self.handle_qpos_addr = self.sim.model.get_joint_qpos_addr(self.doors[self.task_id // 2].joints[1])

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
                # We know we're only setting a single object (the door), so specifically set its pose
            for i in range(2):
                door_pos, door_quat, _ = object_placements[self.doors[i].name]
                door_body_id = self.sim.model.body_name2id(self.doors[i].root_body)
                self.sim.model.body_pos[door_body_id] = door_pos
                self.sim.model.body_quat[door_body_id] = door_quat

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

            eef_pos = self._eef_xpos
            door_pos = np.array(self.sim.data.body_xpos[self.object_body_ids["door"]])
            handle_pos = self._handle_xpos
            hinge_qpos = np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

            di["door_pos"] = door_pos
            di["handle_pos"] = handle_pos
            di[pr + "door_to_eef_pos"] = door_pos - eef_pos
            di[pr + "handle_to_eef_pos"] = handle_pos - eef_pos
            di["hinge_qpos"] = hinge_qpos

            di['object-state'] = np.concatenate([
                di["door_pos"],
                di["handle_pos"],
                di[pr + "door_to_eef_pos"],
                di[pr + "handle_to_eef_pos"],
                di["hinge_qpos"]
            ])

            # Also append handle qpos if we're using a locked door version with rotatable handle
            if self.use_latch:
                handle_qpos = np.array([self.sim.data.qpos[self.handle_qpos_addr]])
                di["handle_qpos"] = handle_qpos
                di['object-state'] = np.concatenate([di["object-state"], di["handle_qpos"]])

        return di

    def _check_success(self):
        """
        Check if door has been opened.
        Returns:
            bool: True if door has been opened
        """
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        if self.task_id % 2 == self.task_id // 2:
                return hinge_qpos > 0.4
        else:
            return hinge_qpos < -0.4

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the door handle.
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the door handle
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.doors[0].important_sites["handle"],
                target_type="site"
            )

    @property
    def _handle_xpos(self):
        """
        Grabs the position of the door handle handle.
        Returns:
            np.array: Door handle (x,y,z)
        """
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _gripper_to_handle(self):
        """
        Calculates distance from the gripper to the door handle.
        Returns:
            np.array: (x,y,z) distance between handle and eef
        """
        return self._handle_xpos - self._eef_xpos


class PandaDoor(DoorEnv):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        if task_id is None:
            task_id = np.random.randint(0, 4)
        super().__init__(robots=['Panda'], task_id=task_id, **kwargs)

class SawyerDoor(DoorEnv):
    """
    Easier version of task - place one object into its bin.
    A new object is sampled on every reset.
    """

    def __init__(self, task_id=None, **kwargs):
        if task_id is None:
            task_id = np.random.randint(0, 4)
        super().__init__(robots=['Sawyer'], task_id=task_id, **kwargs)

if __name__ == '__main__':
    from robosuite.controllers import load_controller_config

    controller = load_controller_config(default_controller="IK_POSE")
    env = SawyerDoor(has_renderer=True, controller_configs=controller, has_offscreen_renderer=False,
                                    reward_shaping=False,
                                    use_camera_obs=False, camera_heights=320, camera_widths=320, render_camera='frontview', task_id=1)
    env.reset()
    for i in range(10000):
        low, high = env.action_spec
        action = np.random.uniform(low=low, high=high)
        env.step(action)
        env.render()