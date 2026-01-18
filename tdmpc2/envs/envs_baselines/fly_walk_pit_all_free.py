import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from gymnasium import utils
from envs.envs_baselines.mujoco_env_gym import MujocoEnv
from envs.envs_baselines.transformation import quaternion_matrix
import mujoco
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary
from PIL import Image
from envs.envs_baselines.generate_terrain import TerrainGenerator

TYPES = {
    "lock design": 0,
    "pos design": 1,
    "execution": 2,
}

class FlyWalkPitAllFreeEnv(MujocoEnv, utils.EzPickle):
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
        
        self.frame_skip = 10
        self.control_frames = 25
        self.env_name = env_name
        default_xml_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "xml_baselines"))
        xml_dir = os.environ.get("TDMPC2_BASELINE_XML_DIR", default_xml_dir)
        xml_path = os.path.join(xml_dir, f"{env_name}.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        xml_string = open(xml_path, 'r').read()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qpos[0] = -20
        random_ang = np.pi / 2
        quaternion = np.array([np.cos(random_ang / 2), 0, 0, np.sin(random_ang / 2)])
        self.init_qpos[3:7] = quaternion
        self.init_qvel = self.data.qvel.ravel().copy()
        self.max_heights = np.zeros(600)
        self.min_heights = np.zeros(600)
        
        # Action space
        self.control_nsteps = 0
        self.control_count = 0
        self.max_episode_steps = 1000

        self.actuator_masks = None
        self.init_joint_ranges = []
        self.render_folder = render_folder

        self.worker_id = worker_id

        self.terrain_generator = TerrainGenerator(xml_string)
        self.ceils_start_pos = []
        self.ceils_end_pos = []
        self.mounts_start_pos = []
        self.mounts_end_pos = []
        self.mounts_heights = []
        self.max_heights = np.ones(600) * 0.25
        self.min_heights = np.ones(600) * 1.4

        MujocoEnv.__init__(self, xml_path, self.frame_skip)
        utils.EzPickle.__init__(self)
        # Observation space: Dict with keys {"type": Discrete(3), "obs": Box}
        sim_dim = int(self.data.qpos.size + self.data.qvel.size)
        terrain_dim = int(self.get_hfield_array().size)
        obs_size = sim_dim + terrain_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.ctrl_cost_coeff = settings.ctrl_cost_coeff_reconfig if settings is not None else 0
        # Action space
        self.action_space = Box(low=-1, high=1, shape=(self.model.nu,), dtype=np.float32)
        self.current_steps = 0

    def update_steps(self, current_steps, total_steps):
        self.current_steps = current_steps

    def sim(self, ctrl):
        ctrl_cost_coeff = self.ctrl_cost_coeff
        self.control_nsteps += 1
        self.control_count += 1
        xposbefore = self.get_body_com("0")[0]
        zposbefore = self.get_body_com("0")[2]
        try:
            collision = self.do_simulation(ctrl, self.frame_skip, True)
        except:
            return 0, True, False, {'use_transform_action': False, 'stage': 'execution'}

        xposafter = self.get_body_com("0")[0]
        zposafter = self.get_body_com("0")[2]
        xpos_index = int(np.clip(self.data.qpos[0] + 300, 0, 599))
        max_height = 10
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_up = (zposafter - zposbefore) / self.dt * 0.8
        if 50 < xposafter < 60 and zposafter > 0.2:
            reward_up *= -1
            if xposafter > 52:
                reward_fwd = 0
        elif 2 < xposafter < 10 or zposafter < 0:
            reward_up *= 1
        else:
            reward_up *= 0

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
        reward = reward_fwd + reward_up + reward_ctrl
        penalty = 0.0

        if self.render_folder is not None:
            frame = self.render(mode='rgb_array')
            img = Image.fromarray(frame)
            img.save(f'{self.render_folder}/%04d.png' % self.control_nsteps)

        min_height = -5
        max_ang = 1.2 * np.pi / 2
        max_x_ang = np.pi / 1.8
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
        if collision:
            print("Collision during simulation!")
            termination = True
            penalty += 20
        truncation = not (self.control_nsteps < max_nsteps)
        return reward, penalty, termination, truncation, energy

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(), 0, False, False, {'use_transform_action': False, 'stage': 'execution'}
        
        total_reward = 0
        total_energy = 0
        termination = False
        truncation = False    
        ctrl = a.copy()
        reward, penalty, t, tr, energy = self.sim(ctrl)
        total_reward += reward
        total_energy += energy
        termination = termination or t
        truncation = truncation or tr
        return self._get_obs(), total_reward, termination, truncation, {
            'use_transform_action': False,
            'energy': total_energy,
        }
    
    def get_hfield_array(self):
        xpos = self.data.qpos[0]
        start_index = int(np.clip(int(xpos) + 295, 0, 600 - 25))
        end_index = int(np.clip(int(xpos) + 295 + 25, 25, 600))
        min_height = self.min_heights[start_index:end_index]
        max_height = self.max_heights[start_index:end_index]
        return np.concatenate([min_height, max_height]).astype(np.float32)

    def _get_obs(self):
        position = self.data.qpos
        velocity = self.data.qvel
        sim_obs = np.concatenate((position, velocity)).astype(np.float32)
        terrain_obs = self.get_hfield_array()
        vec = np.concatenate([sim_obs, terrain_obs]).astype(np.float32)
        return vec

    def reset_state(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[0] = -50
        self.initial_choice_idx = 1
        qpos[1] = self.np_random.uniform(low=-10, high=10, size=1)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.stage = 0
        self.episode_steps = 0
        self.data.eq_active[:] = 0  

    def reset_model(self):
        self.control_nsteps = 0
        self.control_count = 0
        self.reset_state()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def viewer_setup(self):
        pass     
        