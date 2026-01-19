import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

#from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class KHREnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = 136
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = 'cuda'

        self.simulate_action_latency = True  # there is a 1 latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device='cuda', dtype=torch.float32)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device='cuda', dtype=torch.float32)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device='cuda', dtype=torch.float32)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device='cuda', dtype=torch.float32)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device='cuda', dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device='cuda', dtype=torch.float32)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device='cuda', dtype=torch.float32)

        self.rew_buf = torch.zeros((self.num_envs,), device='cuda', dtype=torch.float32)
        self.reset_buf = torch.ones((self.num_envs,), device='cuda', dtype=torch.int32)
        self.episode_length_buf = torch.zeros((self.num_envs,), device='cuda', dtype=torch.int32)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device='cuda', dtype=torch.float32)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device='cuda',
            dtype=torch.float32,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device='cuda', dtype=torch.float32)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        

        
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device='cuda', dtype=torch.float32)
        self.base_quat = torch.zeros((self.num_envs, 4), device='cuda', dtype=torch.float32)
        
        #linkのワールド座標取得用のリスト初期化
        
        #右上半身
        self.r_shoulder_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        #左上半身
        self.l_shoulder_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        #右足
        self.r_hipjointupper_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.r_hipjointlower_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.r_upperleg_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.r_lowerleg_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.r_ankle_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.r_foot_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.r_foot_orientation = torch.zeros((self.num_envs,4),device='cuda', dtype=torch.float32)
        #左足
        self.l_hipjointupper_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.l_hipjointlower_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.l_upperleg_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.l_lowerleg_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.l_ankle_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        self.l_foot_pos = torch.zeros((self.num_envs,3),device='cuda', dtype=torch.float32)
        
                
        self.base_quat = torch.zeros((self.num_envs, 4), device='cuda', dtype=torch.float32)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device='cuda',
            dtype=torch.float32,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), 'cuda')
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), 'cuda')
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), 'cuda')


    def get_observations(self):
        if self.env_cfg["privileged_obs"]:
            self.extras["observations"]["critic"] = self.privileged_obs_buf

        else:
            self.extras["observations"]["critic"] = self.obs_buf
        self.extras["observations"]["dof_pos"] = self.dof_pos.clone()
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device='cuda'))
        return self.obs_buf, None

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
        #res = torch.zeros(self.num_envs, dtype=torch.float, device=gs.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_gait_swing(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device='cuda')
        #res = torch.zeros(self.num_envs, dtype=torch.float, device=gs.device)
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
    
    def _reward_feet_swing_height(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3],dim=2) > 1.0
        pos_error = torch.square(self.feet_pos[:, :, 2] - self.reward_cfg["feet_height_target"]) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_orientation(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_ang_vel_xy(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        
    
    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_joint_torques(self):

        torques = self.robot.get_dofs_control_force()
        return torch.sum(torch.square(torques), dim = 1)

    def _reward_dof_vel(self):
            # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
            # Under BSD-3 License
            return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_acceleration(self):

        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_hip_pos(self):
        # Function borrowed from https://github.com/unitreerobotics/unitree_rl_gym
        # Under BSD-3 License
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    
    def _reward_default_joint_pos(self):

        joint_diff = self.dof_pos - self.default_dof_pos
        left_yaw_roll = joint_diff[:,:2]
        right_yaw_roll = joint_diff[:,6:8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)

        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
    
    def _reward_feet_clearance(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3],dim=2) > 1.0

        feet_z = self.feet_pos[:,:,2]
        delta_z = feet_z - self.last_feet_z

        self.feet_height += delta_z
        self.last_feet_z = feet_z

        '''
        swing_mask = 1 - self._get_gait_phase()

         # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)

        #----------------------------------------------------------------------------------------
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] >= 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1

            rew_pos = torch.abs(self.feet_height - self.reward_cfg["feet_height_target"]) < 0.01
            rew_pos = torch.sum(rew_pos * is_stance)
        '''

        is_stance = self.leg_phase[:, :] >= 0.55
        contact = self.contact_forces[:, self.feet_indices, 2] > 1
        rew_pos = torch.abs(self.feet_height - self.reward_cfg["feet_height_target"]) < 0.005
        rew_pos = torch.sum(rew_pos * is_stance, dim=1)
