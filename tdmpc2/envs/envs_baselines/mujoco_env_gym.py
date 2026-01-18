from collections import OrderedDict
import time

from gymnasium import error, spaces
from gymnasium.utils import seeding
import numpy as np
import gymnasium as gym

try:
    import mujoco
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 1000

from envs.envs_baselines.mjviewer import MjViewer



def convert_observation_to_space(observation):
    if isinstance(observation, list):
        return None

    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, fullpath, frame_skip, mujoco_xml=None):
        self.frame_skip = frame_skip
        self.viewer = None
        self._viewers = {}

        self.is_inited = True
        self.auto_fix_dofs = None
        self.auto_fix_qpos_addrs = None
        self.auto_fix_qvel_addrs = None
        self.gears = None
        self.auto_fix_mass = None
        self.kp = None
        self.kd = None

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reload_sim_model(self, xml_str):
        del self.data
        del self.model
        del self.viewer
        del self._viewers
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.viewer = None
        self._viewers = {}

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        info = {}
        return ob, info

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames, collision_check=False, auto_fix=False, collision_checks=[], collision_check_freq=1):
        self.data.ctrl[:] = ctrl
        collision = False
        for i in range(n_frames):
            if auto_fix:
                qpos = self.data.qpos[self.auto_fix_qpos_addrs]
                qvel = self.data.qvel[self.auto_fix_qvel_addrs]
                fix_torque = -self.kp * qpos - self.kd * qvel
                self.data.ctrl[self.auto_fix_dofs] = fix_torque
            mujoco.mj_step(self.model, self.data)
            if collision_check and i % collision_check_freq == 0:
                if len(collision_checks) == 0:
                    if self.ceil_collision_check() or self.wall_collision_check():
                        collision = True
                        break
                else:
                    for check in collision_checks:
                        if check():
                            collision = True
                            break
        return collision
    
    def set_auto_fix(self, auto_fix_para_dict):
        self.auto_fix_dofs = auto_fix_para_dict.get('auto_fix_dofs', [])
        joint_ids = self.model.actuator_trnid[self.auto_fix_dofs, 0]
        self.auto_fix_qpos_addrs = self.model.jnt_qposadr[joint_ids]
        self.auto_fix_qvel_addrs = self.model.jnt_dofadr[joint_ids]
        self.gears = self.model.actuator_gear[self.auto_fix_dofs, 0]
        self.auto_fix_mass = auto_fix_para_dict.get('auto_fix_mass', 1.0)
        self.kp = 4 * np.pi * np.pi * self.auto_fix_mass / (0.03 * 0.03) / self.gears
        self.kd = 2 * 1.5 * np.sqrt(self.kp * self.gears * self.auto_fix_mass) / self.gears
    
    def joint_qpos(self, i):
        joint_id = self.model.actuator_trnid[i, 0]
        qpos_address = self.model.jnt_qposadr[joint_id]
        qpos = self.data.qpos[qpos_address]
        return qpos
    
    def joint_qvel(self, i):
        joint_id = self.model.actuator_trnid[i, 0]
        qvel_address = self.model.jnt_dofadr[joint_id]
        qvel = self.data.qvel[qvel_address]
        return qvel

    def render(self,
               mode='rgb_array',
               width=DEFAULT_SIZE,
               height=DEFAULT_SIZE,
               camera_id=None,
               camera_name=None):
        if mode == 'rgb_array' or mode == 'depth_array':
            if camera_id is not None and camera_name is not None:
                raise ValueError("Both `camera_id` and `camera_name` cannot be"
                                 " specified at the same time.")

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'track'

            if camera_id is None and camera_name in [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]:
                camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        if mode == 'rgb_array':
            # Create renderer if needed
            renderer = self._get_viewer(mode)
            renderer.update_scene(self.data, camera=camera_id if camera_id is not None else -1)
            return renderer.render()
        elif mode == 'depth_array':
            # Create renderer if needed
            renderer = self._get_viewer(mode)
            renderer.update_scene(self.data, camera=camera_id if camera_id is not None else -1)
            return renderer.render()
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = MjViewer(self.model, self.data)
            elif mode == 'rgb_array':
                self.viewer = mujoco.Renderer(self.model, height=DEFAULT_SIZE, width=DEFAULT_SIZE)
            elif mode == 'depth':
                self.viewer = mujoco.Renderer(self.model, height=DEFAULT_SIZE, width=DEFAULT_SIZE)
                self.viewer.enable_depth_rendering()

            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def set_custom_key_callback(self, key_func):
        self._get_viewer('human').custom_key_callback = key_func

    def get_body_com(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id]

    def state_vector(self):
        return np.concatenate([
            self.data.qpos.flat,
            self.data.qvel.flat
        ])

    def vec_body2world(self, body_name, vec):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_xmat = self.data.xmat[body_id].reshape(3, 3)
        vec_world = (body_xmat @ vec[:, None]).ravel()
        return vec_world

    def pos_body2world(self, body_name, pos):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_xpos = self.data.xpos[body_id]
        body_xmat = self.data.xmat[body_id].reshape(3, 3)
        pos_world = (body_xmat @ pos[:, None]).ravel() + body_xpos
        return pos_world
    
    def ceil_collision_check(self):
        geom_ceil_ids = []
        for i in range(100):
            geom_ceil_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"ceil_{i}")
            if geom_ceil_id == -1:
                break
            geom_ceil_ids.append(geom_ceil_id)
        collision = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for geom_ceil_id in geom_ceil_ids:
                if con.geom1 == geom_ceil_id or con.geom2 == geom_ceil_id:
                    collision = True
                    break
            if collision:
                break
        return collision

    def wall_collision_check(self):
        i = 0
        geom_wall_ids = []
        for i in range(100):
            geom_wall_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{i}")
            if geom_wall_id == -1:
                break
            geom_wall_ids.append(geom_wall_id)
        collision = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for geom_wall_id in geom_wall_ids:
                if con.geom1 == geom_wall_id or con.geom2 == geom_wall_id:
                    collision = True
                    break
            if collision:
                break
        return collision
