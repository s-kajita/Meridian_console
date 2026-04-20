import torch
import math
import genesis as gs
from tensordict import TensorDict
import numpy as np
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

#from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class KHREnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True):
        self.num_envs: int = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales: dict[str, float] = obs_cfg["obs_scales"]
        self.reward_scales: dict[str, float] = reward_cfg["reward_scales"]

        self.obs_buf = torch.zeros((self.num_envs, 71),dtype=torch.float32,device='cuda')
        self.privileged_obs_buf = torch.zeros((self.num_envs, 50),dtype=torch.float32,device='cuda')
        

       

        #ロータイナーシャの設定 set armature:  default = 0.1 kgm^2
        #self.robot.set_dofs_armature(0.01)


        # Define global gravity direction vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda')

        # Initial state
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=torch.float32, device='cuda')
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=torch.float32, device='cuda')
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # initialize buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=torch.float32, device='cuda')
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=torch.float32, device='cuda')
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=torch.float32, device='cuda')
        self.rew_buf = torch.empty((self.num_envs,), dtype=torch.float32, device='cuda')
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.bool, device='cuda')
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=torch.int32, device='cuda')
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=torch.float32, device='cuda')
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device='cuda',
            dtype=torch.float32,
        )
        self.commands_limits: tuple[torch.Tensor, torch.Tensor] = tuple(
            torch.tensor(values, dtype=torch.float32, device='cuda')
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device='cuda')
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=torch.float32, device='cuda')
        self.base_quat = torch.empty((self.num_envs, 4), dtype=torch.float32, device='cuda')
        self.base_euler = torch.empty((self.num_envs, 3), dtype=torch.float32, device='cuda')
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=torch.float32,
            device='cuda',
        )

        self.extras = dict()  # extra information for logging

        #各joint,linkのワールド座標取得変数の初期化
        self.l_ankle_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)

        self.r_ankle_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)

        #歩行周期初期化
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.phase_left = torch.zeros(self.num_envs, device=self.device)
        self.phase_right = torch.zeros(self.num_envs, device=self.device)

        self.leg_phase = torch.zeros((self.num_envs, 2), device=self.device)
        self.sin_phase = torch.zeros((self.num_envs, 1), device=self.device)
        self.cos_phase = torch.zeros((self.num_envs, 1), device=self.device)

        self.feet_height_sharpness = 50
        self.target_feet_height = self.reward_cfg["feet_height_target"]

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=torch.float32, device='cuda')


    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)

    
    def get_observations(self):
        #return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])
        return TensorDict({"policy": self.obs_buf, "privileged":self.privileged_obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        # reset state

        # reset buffers
        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        # fill extras
        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)
        #domain randomization
        if self.env_cfg['randomize_base_mass']:
            self._randomize_mass(envs_idx)
        if self.env_cfg['randomize_friction']:
            self._randomize_friction(envs_idx)
        # random sample command upon reset
        self._resample_commands(envs_idx)

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.cos_phase, #1
                self.sin_phase, #1
                self.last_actions, #12
                self.last_dof_vel * self.obs_scales["dof_vel"],  # 12
            ),
            dim=-1,
        )
        
        self.privileged_obs_buf = torch.cat(
            [
                #特権情報のみ
                #ノイズのないセンサの値
                #摩擦係数
                self._added_base_mass
            ],
            dim=-1,
        )
        

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()
    
    #------------- domain randomization------------
    '''
    def _resample_commands(self, envs_idx):
        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        if envs_idx is None:
            self.commands.copy_(commands)
        else:
            torch.where(envs_idx[:, None], commands, self.commands, out=self.commands)
    '''


    def _randomize_friction(self, env_ids):
        '''
        ratios = gs_rand(*self.friction_range, (self.num_envs, self.robot.n_links))
        if env_ids is None:
            self._friction_value.copy_(ratios)
            self.robot.set_friction_ratio(self._friction_value, range(self.robot.n_links), env_ids)
        
        else:
            #torch.where(env_ids[:, None], ratios, self._friction_value, out=self._friction_value)
            self._friction_value[env_ids] = ratios[env_ids]
            self.robot.set_friction_ratio(self._friction_value, range(self.robot.n_links), env_ids)
        '''
        #要修正
        min_mass, max_mass = self.env_cfg['mass_range']

        if env_ids is None:

            added_mass = gs.rand((self.num_envs, 1), dtype=float) * (max_mass - min_mass) + min_mass
            self._added_base_mass.copy_(added_mass)
            self.robot.set_mass_shift(added_mass,[self.baselink_id],None)
        else:
            env_idx = env_ids.nonzero(as_tuple=False).flatten()
            if len(env_idx) == 0:
                return  # 何もリセットされてない
            added_mass = gs.rand((len(env_idx), 1), dtype=float) * (max_mass - min_mass) + min_mass
            self._added_base_mass[env_idx] = added_mass
            self.robot.set_mass_shift(added_mass,[self.baselink_id],env_idx)

    

    def _randomize_mass(self, env_ids):
        '''
        動作確認済みOK
        min_mass, max_mass = self.env_cfg['mass_range']
        print(env_ids)
        added_mass = gs.rand((self.num_envs, 1), dtype=float) * (max_mass - min_mass) + min_mass
        self._mass_value = added_mass
        self.robot.set_mass_shift(added_mass, [self.baselink_id,])
        '''
        min_mass, max_mass = self.env_cfg['mass_range']

        if env_ids is None:

            added_mass = gs.rand((self.num_envs, 1), dtype=float) * (max_mass - min_mass) + min_mass
            self._added_base_mass.copy_(added_mass)
            self.robot.set_mass_shift(added_mass,[self.baselink_id],None)
        else:
            env_idx = env_ids.nonzero(as_tuple=False).flatten()
            if len(env_idx) == 0:
                return  # 何もリセットされてない
            added_mass = gs.rand((len(env_idx), 1), dtype=float) * (max_mass - min_mass) + min_mass
            self._added_base_mass[env_idx] = added_mass
            self.robot.set_mass_shift(added_mass,[self.baselink_id],env_idx)
            
            



    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_alive(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        return 1.0
    
    def _reward_gait_contact(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        res = torch.zeros(self.num_envs, dtype=torch.float, device='cuda')
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_gait_swing(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device='cuda')
        for i in range(self.feet_num):
            is_swing = self.leg_phase[:, i] >= 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_swing)
        return res
    
    def _reward_contact_no_vel(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3],
                             dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
    
    def _reward_feet_clearance(self):
        
        is_swing = self.leg_phase[:, :] >= 0.55
        error = torch.abs(self.target_feet_height - self.feet_pos[:, :, 2])
        pos = torch.exp(-self.feet_height_sharpness * error)
        #pos = torch.exp(-self.feet_height_sharpness * (torch.abs(self.target_feet_height - self.feet_height )))
        rew = torch.sum(pos * is_swing, dim=1)
            
        return rew
    
    def _reward_hip_pos(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        #return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
        return torch.sum(torch.square(self.dof_pos[:,[1,7]]), dim=1)
    
    def _reward_orientation(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler[:, :2]), dim=1) * 10)
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)

        penalty = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        reward = (quat_mismatch + orientation) / 2

        return penalty
    
    def _reward_ang_vel_xy(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_joint_torques(self):

        torques = self.robot.get_dofs_control_force()
        return torch.sum(torch.square(torques), dim = 1)
    
    def _reward_dof_vel(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_acceleration(self):

        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_collision(self):

        return 0
