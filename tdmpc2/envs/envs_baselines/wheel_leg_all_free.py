import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from gymnasium import utils
from envs.envs_baselines.mujoco_env_gym import MujocoEnv
from envs.envs_baselines.transformation import quaternion_matrix
import mujoco
from gymnasium.spaces import Box
from PIL import Image
from envs.envs_baselines.generate_terrain import StairsGenerator
from envs.envs_baselines.motor import MotorModel2208

class WheelLegAllFreeEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(self, env_name="fly_plane", render_folder=None, worker_id=-1,
                 settings=None,
                 **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        
        self.frame_skip = 100
        self.control_frames = 5
        self.env_name = env_name
        default_xml_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xml_baselines"))
        xml_dir = os.environ.get("TDMPC2_BASELINE_XML_DIR", default_xml_dir)
        xml_path = os.path.join(xml_dir, f"{env_name}.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.initialize_state()
        self.viewer = None
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        self.start = 0
        self.tunnel_start = 0
        self.end = 0
        self.start_1 = 0
        self.end_1 = 0
        self.stair_height = 0.05
        self.hfield_dict = {}
        self.height_sample_unit = 0.05
        
        # Action space
        self.control_nsteps = 0
        self.control_count = 0
        self.max_episode_steps = 1000

        self.actuator_masks = None
        self.init_joint_ranges = []
        self.render_folder = render_folder

        self.worker_id = worker_id

        MujocoEnv.__init__(self, xml_path, self.frame_skip)
        utils.EzPickle.__init__(self)
        # Observation space
        obs_size = self.data.qpos[:].shape[0] + self.data.qvel[:].shape[0]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.ctrl_cost_coeff = settings.ctrl_cost_coeff_reconfig if settings is not None else 0
        self.stair_generator = StairsGenerator(open(xml_path, 'r').read())
        self.motor_model = MotorModel2208()
        low_bound = np.ones(12, dtype=np.float32) * -2.5
        high_bound = np.ones(12, dtype=np.float32) * 2.5
        low_bound[6:8] = -0.25
        low_bound[:2] = -0.25
        high_bound[2:6] = 0.25
        low_bound[8:] = -1
        high_bound[8:] = 1
        self._raw_action_low = low_bound
        self._raw_action_high = high_bound
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        for i in range(12):
            self.data.eq_active[i] = 0

        # Collision check
        self.geom_ids = set()

    def initialize_state(self):
        self.data.ctrl[:8] = 1.5
        self.data.ctrl[2:6] *= -1
        for _ in range(20000):
            mujoco.mj_step(self.model, self.data)
        ang = - np.pi / 2
        quaternion = np.array([np.cos(ang / 2), 0, 0, np.sin(ang / 2)])
        self.data.qpos[3:7] = quaternion

    def update_steps(self, current_steps, total_steps):
        pass

    def collision_check(self):
        collision = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 in self.geom_ids or con.geom2 in self.geom_ids:
                collision = True
                break
        return collision


    def sim(self, ctrl, auto_fix=False):
        ctrl_cost_coeff = self.ctrl_cost_coeff
        self.control_nsteps += 1
        self.control_count += 1
        xposbefore = self.get_body_com("0")[0]
        try:
            collision = self.do_simulation(ctrl, self.frame_skip, True, auto_fix, [self.collision_check], collision_check_freq=5)
        except Exception as e:
            print(e)
            return 0, True, False, 0

        xposafter = self.get_body_com("0")[0]
        xindex = int(np.clip(int(xposafter * 2) + 600, 0, 1199))
        
        reward_fwd = (xposafter - xposbefore) / self.dt

        s = self.state_vector()
        height = s[2]
        zdir = quaternion_matrix(s[3:7])[:3, 2]
        xdir = quaternion_matrix(s[3:7])[:3, 0]
        ang = np.arccos(zdir[2])
        x_ang = np.abs(np.arccos(xdir[0]) - np.pi / 2)
        omega = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            qvel_address = self.model.jnt_dofadr[joint_id]
            omega[i] = self.data.qvel[qvel_address]
        energy = np.dot(omega, ctrl * self.model.actuator_gear[:, 0]) * self.dt

        reward_ctrl = - ctrl_cost_coeff * energy
        reward = reward_fwd + reward_ctrl

        if self.render_folder is not None:
            frame = self.render(mode='rgb_array')
            img = Image.fromarray(frame)
            img.save(f'{self.render_folder}/%04d.png' % self.control_nsteps)

        min_height = -0.5
        max_height = 15
        max_ang = np.pi / 2
        max_nsteps = 1000
        termination = False
        if not np.isfinite(s).all():
            print("State not finite!")
            print("Stats:", s)
            termination = True
        if height < min_height:
            print("Height too low!")
            print("height: ", height)
            print()
            termination = True
        if height > max_height:
            print("Height too high!")
            print("height: ", height)
            termination = True
        if ang >= max_ang:
            print("Angle out of bound!")
            print("angle: ", ang)
            termination = True
        if x_ang >= max_ang:
            print("X Angle out of bound!")
            print("x_angle: ", x_ang)
            termination = True
        if collision:
            print("Collision during simulation!")
            termination = True
        truncation = not (self.control_nsteps < max_nsteps)
        return reward, termination, truncation, energy

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(), 0, False, False, {'use_transform_action': False, 'stage': 'execution'}

        total_reward = 0
        total_energy = 0
        termination = False
        truncation = False    
        ctrl = self._denormalize_action(a)
        reward, t, tr, energy = self.sim(ctrl)
        x_angle = np.abs(np.arccos(quaternion_matrix(self.data.qpos[3:7])[:3, 0][0]) - np.pi / 2)
        reward -= x_angle * 0.01
        # reward -= np.sum(np.square(np.clip(a + 0.1 - self.high_bound, 0, 0.1))) * 2
        # reward -= np.sum(np.square(np.clip(self.low_bound + 0.1 - a, 0, 0.1))) * 2
        total_reward += reward
        total_energy += energy
        termination = termination or t
        truncation = truncation or tr
        if self.data.qpos[0] > self.end_1 + 5:
            termination = True
        return self._get_obs(), total_reward, termination, truncation, {
            'use_transform_action': False,
            'energy': total_energy,
        }

    def _denormalize_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        scale = (self._raw_action_high - self._raw_action_low) * 0.5
        bias = (self._raw_action_high + self._raw_action_low) * 0.5
        return action * scale + bias
    
    def torque_limit(self, ctrl):
        motor_ctrl = ctrl[8:] * self.model.actuator_gear[8:, 0]
        motor_omega = self.joint_qvel([8, 9, 10, 11])
        clipped_motor_ctrl = self.motor_model.apply_limit(motor_ctrl, motor_omega)
        ctrl[8:] = clipped_motor_ctrl / self.model.actuator_gear[8:, 0]
        return ctrl
    

    def _get_obs(self):
        position = self.data.qpos
        velocity = self.data.qvel
        sim_obs = np.concatenate((position, velocity)).astype(np.float32)
        return sim_obs
    
    def reset_collision(self):
        self.geom_ids = set()
        for i in range(100):
            id_0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"collision_{i}_0")
            id_1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"collision_{i}_1")
            if id_0 == -1 or id_1 == -1:
                break
            self.geom_ids.add(id_0)
            self.geom_ids.add(id_1)

    def reset_state(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        random_ang = - np.pi / 2
        quat = np.array([np.cos(random_ang / 2), 0, 0, np.sin(random_ang / 2)])
        self.data.qpos[3:7] = quat
        self.data.qpos[0] = -38.518
        for i in range(12):
            self.data.eq_active[i] = 0
        self.stage = 0 

    def reset_model(self):
        self.control_nsteps = 0
        self.control_count = 0
        self.reset_state()
        self.reset_collision()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def viewer_setup(self):
        pass
        