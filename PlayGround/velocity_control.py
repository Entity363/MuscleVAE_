import argparse
from MuscleVAECore.Env.muscle_env import MuscleEnv
from MuscleVAECore.Model.musclevae import MuscleVAE
from MuscleVAECore.Utils.misc import load_data, load_yaml
from MuscleVAECore.Utils.motion_utils import state2ob
from MuscleVAECore.Utils.pytorch_utils import build_mlp
from MuscleVAECore.Utils.radam import RAdam
from PlayGround.playground_util import get_root_facing, state2speed
from random_generation import RandomPlayground
import MuscleVAECore.Utils.pytorch_utils as ptu
import torch
import numpy as np
import types
from scipy.spatial.transform import Rotation
import psutil
import MuscleVAECore.Utils.pytorch_utils as ptu

import math
import keyboard

from mpi4py import MPI
from collections import deque

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def random_target(env):
    speed = np.random.choice(env.speed_range)
    direction_angle = np.random.uniform(0, np.pi * 2)
    res = np.array([speed, direction_angle])
    return res

def speed_target(self):
    if not hasattr(self, 'target') or self.target is None:
        self.target = random_target(self)
    return self.target


def hand_control_wasd():
    """
    Read WASD keys and return angle and velocity norm.
    Angle is in radians, 0 = facing right (x+), counterclockwise positive.
    Velocity norm is the speed magnitude (0 if no keys pressed).
    """

    # Directions mapped to x,y
    x_dir = 0
    y_dir = 0

    running = False

    if keyboard.is_pressed('w'):
        y_dir += 1
    if keyboard.is_pressed('s'):
        y_dir -= 1
    if keyboard.is_pressed('a'):
        x_dir += 1
    if keyboard.is_pressed('d'):
        x_dir -= 1
    if keyboard.is_pressed('shift'):
        running = True
    else:
        running = False

    # If no input, no movement
    if x_dir == 0 and y_dir == 0:
        return 0, 0

    # Calculate angle from vector
    angle = math.atan2(y_dir, x_dir)

    # Normalize velocity magnitude (can be scaled)
    velo_norm = math.sqrt(x_dir**2 + y_dir**2)
    if running == True:
        max_speed = 3.0  # max speed value you want
    else:
        max_speed = 1.0
    velo_norm = min(velo_norm, 1) * max_speed  # max speed capped at max_speed

    return angle, velo_norm


def hand_control_1_minute(self):
    """
    In drawstuff mode, it is not easy to use the keyboard to control the character.
    """
    if self.interactor.time_step < 400:
        angle = 35.0*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 400 and  self.interactor.time_step < 800:
        angle =  60.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 800 and  self.interactor.time_step < 1200:
        angle =  75.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 1200 and  self.interactor.time_step < 1600:
        angle = 180*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 1600 and  self.interactor.time_step < 2000:
        angle = 180*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 2000 and  self.interactor.time_step < 2400:
        angle = 150*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 2400 and  self.interactor.time_step < 2800:
        angle = 0
        velo_norm = 3.0
    elif  self.interactor.time_step >= 2800 and  self.interactor.time_step < 3200:
        angle = 0
        velo_norm = 2.0
    elif  self.interactor.time_step >= 3200 and  self.interactor.time_step < 3600:
        angle = 60*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 3600 and  self.interactor.time_step < 4000:
        angle =  120.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 4000 and  self.interactor.time_step < 4400:
        angle =  75.0*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 4400 and  self.interactor.time_step < 4800:
        angle = 20*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 4800 and  self.interactor.time_step < 5200:
        angle = 40*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 5200 and  self.interactor.time_step < 5600:
        angle = 150*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 5600 and  self.interactor.time_step < 6000:
        angle = 0
        velo_norm = 3.0
    elif  self.interactor.time_step >= 6000 and  self.interactor.time_step < 6400:
        angle = 60*np.pi/180
        velo_norm = 2.0
    elif  self.interactor.time_step >= 6400 and  self.interactor.time_step < 6800:
        angle = 60*np.pi/180
        velo_norm = 3.0
    elif  self.interactor.time_step >= 6800 and  self.interactor.time_step < 7200:
        angle = 60*np.pi/180
        velo_norm = 2.0
    else:
        angle = -180.0*np.pi/180
        velo_norm = 3.0

    return angle, velo_norm


def after_step(self, **kargs):
    self.step_cnt += 1
    self.target_scaled_muscle_len = self.get_target_scaled_muscle_len()
    if not self.interaction:
        if self.step_cnt % self.random_count == 0:
            self.target = random_target(self)
    else:
        angle, velo_norm = hand_control_wasd() #hand_control_1_minute(self)
        res = np.array([velo_norm, angle])
        self.target = res



def after_substep(self):
    #self.interactor.time_step += 1
    pass


class SpeedPlayground(RandomPlayground):
    def __init__(self,observation_size, observation_rigid_size, action_size, delta_size, env, **kargs):
        kargs['replay_buffer_size'] = 5000
        super().__init__(observation_size, observation_rigid_size, action_size, delta_size, env, **kargs)
        self.observation_size = observation_size
        self.latent_size = kargs['latent_size']
        self.batch_size = 512
        self.collect_size = 500
        self.env.max_length = 512
        self.runner.with_noise = kargs['train'] # use act_determinastic....
        self.time_step = 0
        
        if mpi_rank == 0:
            self.replay_buffer.reset_max_size(20000)
        
        self.show = kargs['show']
        self.env.speed_range = [0,0,1,2,3]
        self.env.get_target = types.MethodType(speed_target, self.env)
       
        self.env.interaction = kargs['interaction']
        self.env.after_step = types.MethodType(after_step, self.env)   
        self.env.after_substep = types.MethodType(after_substep, self.env)
        
        self.env.show = self.show
        self.build_high_level()
        
        self.mass = self.env.sim_character.body_info.mass_val / self.env.sim_character.body_info.sum_mass
        self.mass = ptu.from_numpy(self.mass).view(-1)

        self.dance = kargs['dance']
        self.show_arrow = True

        if self.show:
            if self.mode == 'drawstuff':
                try:
                    from VclSimuBackend.ODESim.Loader.MeshCharacterLoader import MeshCharacterLoader
                except ImportError:
                    import VclSimuBackend
                    MeshCharacterLoader = VclSimuBackend.ODESim.MeshCharacterLoader
                MeshLoader = MeshCharacterLoader(self.env.scene.world, self.env.scene.space)
                self.env.arrow = MeshLoader.load_from_obj('./Data/Misc/drawstuff/arrow.obj', 'arrow', volume_scale=1, density_scale=1)
                self.env.arrow.is_enable = False
                self.env.arrow = self.env.arrow.root_body
                self.env.base_rotation = Rotation.from_rotvec(np.array([-np.pi/2,0,0]))
                #self.env.interactor = self
                



    #-----------------------------deal with parameters--------------------------------#
    def parameters_for_sample(self):
        res =  super().parameters_for_sample()
        res['high_level'] = self.high_level.state_dict()
        return res
    
    def load_parameters_for_sample(self, dict):
        super().load_parameters_for_sample(dict)
        self.high_level.load_state_dict(dict['high_level'])
    
    def try_evaluate(self, iteration):
        pass
    
    def try_save(self, iteration):
        if iteration % self.save_period == 0:
            check_point = {
                    'self': self.state_dict(),
                    'wm_optim': self.wm_optimizer.state_dict(),
                    'vae_optim': self.vae_optimizer.state_dict(),
                    'balance': self.env.val,
                    'high_level': self.high_level.state_dict(),
                    'high_level_optim': self.high_level_optim.state_dict(),
                }
            import os
            print("saved on velocity")
            torch.save(check_point, os.path.join(self.data_dir_name,f'{iteration}.data'))

    def try_load(self, data_file):
        data = super().try_load(data_file)
        self.high_level.load_state_dict(data['high_level'])
        return data
    @property
    def dir_prefix(self):
        return 'Experiment/playground'
    
    def cal_rwd(self, **obs_info):
        return 0
    
    #------------------------------------------task-----------------------------------#    
    @property   
    def task_ob_size(self):
        return self.observation_size + 3
    
    def build_high_level(self):
        self.high_level = build_mlp(self.task_ob_size, self.latent_size, 3, 256, 'ELU').to(ptu.device)
        self.high_level_optim = RAdam(self.high_level.parameters(), lr=1e-3)
        lr = lambda epoach: max(0.99**(epoach), 1e-1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.high_level_optim, lr)
        
    @staticmethod
    def target2n_target(state, target):
        if len(state.shape) ==2:
            state = state[None,...]
        if len(target.shape) ==1:
            target = target[None,...]
        if isinstance(target, np.ndarray):
            target = ptu.from_numpy(target)
        if isinstance(state, np.ndarray):
            state = ptu.from_numpy(state)
        facing_direction = get_root_facing(state)
        facing_angle = torch.arctan2(facing_direction[:,2], facing_direction[:,0])
        delta_angle = target[:,1] - facing_angle
        res = torch.cat([target[:,0, None], torch.cos(delta_angle[:,None]), torch.sin(delta_angle[:,None])], dim = -1)
        return res
        
    
    #------------------------------------------acting-------------------------------#
    def act_task_old(self, **obs_info):
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, _ = self.encoder.encode_prior(n_observation)    
        n_target = self.target2n_target(obs_info['state'], obs_info['target'])
        
        task = torch.cat([n_observation, n_target], dim=1)
        offset = self.high_level(task)
        if self.dance:
            if n_target[...,2].abs()<0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu+offset
        
        action = self.decode(n_observation, latent)
        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset
        }
    #------------------------------------------acting-------------------------------#
    def act_task_old(self, **obs_info):
        
        n_observation = self.obsinfo2n_obs(obs_info)
        latent, mu, _ = self.encoder.encode_prior(n_observation)    
        n_target = self.target2n_target(obs_info['state'], obs_info['target'])
        
        task = torch.cat([n_observation, n_target], dim=1)
        offset = self.high_level(task)
        if self.dance:
            if n_target[...,2].abs()<0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu+offset
        
        action = self.decode(n_observation, latent)
        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset
        }
    def act_task(self, **obs_info):
        """
        High-level policy for MuscleVAE:
        - input: normalized rigid obs (+ optional muscle_state), + target [speed, heading]
        - output: muscle actions via encoder + high_level + decoder
        """

        # 1) Get normalized rigid observation (371-dim)
        if 'n_observation' in obs_info:
            n_observation = obs_info['n_observation']        # [B, 371] or [1, 371]
        else:
            if 'observation_rigid' in obs_info:
                cur_observation = obs_info['observation_rigid']
            else:
                cur_observation = state2ob(obs_info['state'])
            # For MuscleVAE use normalize_target 
            n_observation = self.normalize_target(cur_observation)

        if isinstance(n_observation, np.ndarray):
            n_observation = ptu.from_numpy(n_observation)
        if n_observation.dim() == 1:
            n_observation = n_observation.unsqueeze(0)       # → [1, 371]
        n_observation = n_observation.to(ptu.device)

        # 2) Append muscle_state if available (to reach 386 dims)
        muscle_state = obs_info.get('muscle_state', None)
        if self.env.use_muscle_state and muscle_state is not None:
            # to tensor
            if isinstance(muscle_state, np.ndarray):
                muscle_state = ptu.from_numpy(muscle_state)
            muscle_state = muscle_state.to(
                n_observation.device, dtype=n_observation.dtype
            )

            # ensure [B, ms_dim]
            if muscle_state.dim() == 1:
                muscle_state = muscle_state.unsqueeze(0)      # [1, ms_dim]

            # fix batch mismatch like [1,ms_dim] vs [B,371]
            if muscle_state.shape[0] != n_observation.shape[0]:
                if muscle_state.shape[0] == 1:
                    muscle_state = muscle_state.expand(
                        n_observation.shape[0], -1
                    )
                else:
                    raise RuntimeError(
                        f"act_task: batch mismatch: n_observation {n_observation.shape}, "
                        f"muscle_state {muscle_state.shape}"
                    )

            enc_inp = torch.cat([n_observation, muscle_state], dim=-1)  # [B, 386]
        else:
            enc_inp = n_observation  # [B, 371] if muscle state is disabled

        # 3) Prior encoder on full obs (386) – matches Linear(386,512)
        latent, mu, _ = self.encoder.encode_prior(enc_inp)

        # 4) Task input: obs + target → 386 + 3 = 389 dims
        n_target = self.target2n_target(
            obs_info['state'], obs_info['target']
        )  # [B, 3]
        if isinstance(n_target, np.ndarray):
            n_target = ptu.from_numpy(n_target).to(enc_inp.device)
        if n_target.dim() == 1:
            n_target = n_target.unsqueeze(0)

        task = torch.cat([enc_inp, n_target], dim=1)  # [B, self.task_ob_size]

        offset = self.high_level(task)
        if self.dance:
            if n_target[..., 2].abs() < 0.5:
                latent = latent
            else:
                latent = latent + offset
        else:
            latent = mu + offset

        # 5) Decode using the same enc_inp that encoder saw
        action = self.decode(enc_inp, latent)

        return action, {
            'mu': mu,
            'latent': latent,
            'offset': offset,
        }



    
    def act_determinastic(self, obs_info):
        return self.act_task(**obs_info)[0]
    
    #------------------------------------------training-------------------------------#
    @property
    def high_level_data_name_list(self):
        return ['state', 'target', 'muscle_state']
    
    
    
    def train_one_step(self):
        
        name_list = self.high_level_data_name_list
        rollout_length = 16
        # self.sub_iter = 2
        data_loader = self.replay_buffer.\
            generate_data_loader(   name_list,
                                    rollout_length,
                                    self.musclevae_batch_size,
                                    self.sub_iter
                                )
        for batch in data_loader:
            log = self.train_high_level(*batch)
        self.scheduler.step()
        return log

    
    def get_loss(self, state, target):
        direction = get_root_facing(state)
        delta_angle = torch.atan2(direction[:, 2], direction[:, 0]) - target[:, 1]
        direction_loss = torch.acos(
            torch.cos(delta_angle).clamp(min=-1+1e-4, max=1-1e-4)
        ) / torch.pi
        
        com_vel = state2speed(state, self.mass)
        target_direction = torch.cat(
            [torch.cos(target[:, 1, None]), torch.sin(target[:, 1, None])],
            dim=-1
        )

        # if speed == 0 use |v|_1, otherwise project on target direction
        com_vel_proj = torch.where(
            target[:, 0] == 0,
            torch.norm(com_vel, dim=-1, p=1),
            torch.einsum('bi,bi->b', com_vel[:, [0, 2]], target_direction)
        )

        speed_loss = torch.abs(com_vel_proj - target[:, 0]) / target[:, 0].clamp(min=1)
        
        fall_down_loss = torch.clamp(state[..., 0, 1], min=0, max=0.6)
        fall_down_loss = (0.6 - fall_down_loss)
        fall_down_loss = torch.mean(fall_down_loss)

        return direction_loss.mean(), speed_loss.mean(), fall_down_loss
    
    def train_high_level(self, states, targets, muscle_states):
        """
        states:        [B, T, state_dim]
        targets:       [B, T, 2]   # [speed, heading]
        muscle_states: [B, T, ms_dim]
        """
        rollout_length = states.shape[1]

        states        = states.to(ptu.device)
        targets       = targets.to(ptu.device)
        muscle_states = muscle_states.to(ptu.device)

        # initial state
        cur_state        = states[:, 0]        # [B, state_dim]
        cur_muscle_state = muscle_states[:, 0] # [B, ms_dim]

        cur_observation = state2ob(cur_state)
        n_observation   = self.normalize_target(cur_observation)  # ← use normalize_obs here

        loss_name = ['direction', 'speed', 'fall_down', 'acs']
        loss_num  = len(loss_name)
        loss      = [[] for _ in range(loss_num)]

        for i in range(rollout_length):
            # ---- 1) High-level action via act_task (same as ControlVAE) ----
            action, info = self.act_task(
                state=cur_state,
                target=targets[:, i],
                n_observation=n_observation,
                muscle_state=cur_muscle_state,
            )

            # ---- 2) World model rollout (muscle-aware) ----
            # NOTE: MuscleVAE world_model returns (state, muscle_state)
            cur_state, cur_muscle_state = self.world_model(
                cur_state,
                action,
                muscle_state=cur_muscle_state
            )

            # ---- 3) Update observation for next step ----
            cur_observation = state2ob(cur_state)
            n_observation   = self.normalize_target(cur_observation)

            # ---- 4) Task losses ----
            d_loss, s_loss, f_loss = self.get_loss(cur_state, targets[:, i])
            loss[0].append(d_loss)
            loss[1].append(s_loss)
            loss[2].append(f_loss)

            # ---- 5) Action regularization on info['offset'] ----
            action_loss = torch.mean(info['offset'] ** 2)
            loss[3].append(action_loss)

        # ---- 6) Temporal averaging + weighting ----
        weight = [1, 0, 100, 20]
        loss_value = [sum(l) / rollout_length * weight[i] for i, l in enumerate(loss)]
        loss_value[0] = loss[0][-1]  # direction: only last step

        total_loss = sum(loss_value)

        # ---- 7) Optimize high_level only ----
        self.high_level_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.high_level.parameters(), 1)
        self.high_level_optim.step()

        res = {loss_name[i]: loss_value[i] for i in range(loss_num)}
        res['loss'] = total_loss
        return res

        
    #------------------------------------------playing-------------------------------#
    def get_action(self, **obs_info):
        return self.act_task(**obs_info)
    

    #-----------------------------------------configuring----------------------------#
    @staticmethod
    def build_arg(parser = None):
        import yaml
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--mode', default = 'drawstuff', type = str)
        parser.add_argument('--show', default = False, action='store_true')
        parser.add_argument('--drawstuff_render_mode', default = 1, type=int)
        parser = MuscleEnv.add_specific_args(parser)
        parser = MuscleVAE.add_specific_args(parser)
        args = vars(parser.parse_args())
        config = load_yaml(initialdir ='Data/NNModel/Pretrained')
        args.update(config)

        return args
       
        
if __name__ == "__main__":
    if mpi_rank ==0:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', default=0, type = int)
        parser.add_argument('--interaction', default=True,  action='store_true')
        parser.add_argument("--use_as_highlevel", type=int, default=1, help = "whether use highlevel train")
        parser.add_argument("--env_step_mode", type=str, default='fatigue_dynamics_constrain', help = "step mode in env")
        parser.add_argument('--gpu', type = int, default=0, help='gpu id')
        parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
        parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
        parser.add_argument('--experiment_name', type = str, default="speed_playground", help="")
        parser.add_argument('--muscle_capacity_ratio', type = float, default=1.0, help='how muscle muscle capacity of your model')
        parser.add_argument('--muscle_max_force_at', type = float, default=1.0, help='how muscle muscle force could be')
    
        args = SpeedPlayground.build_arg(parser)
        import tkinter.filedialog as fd
        data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
    args = mpi_comm.bcast(None if mpi_rank!=0 else args, root=0)
    ptu.init_gpu(True, gpu_id=args['gpu'])
    if args['cpu_e'] !=-1:
        p = psutil.Process()
        cpu_lst = p.cpu_affinity()
        try:
            p.cpu_affinity(range(args['cpu_b'],args['cpu_e']))   
        except:
            pass 

    data_file = mpi_comm.bcast(None if mpi_rank!=0 else data_file, root=0)
    env = MuscleEnv(show_drawstuff=False,**args)
    
    rigid_observation_sz = 371
    muscle_observation_sz =  5 * 3
    observation_sz = rigid_observation_sz + muscle_observation_sz 
    muscle_action = 284
    joint_torque_action = 66
    action_sz = muscle_action 
    playground = SpeedPlayground(observation_sz, rigid_observation_sz, action_sz, 138, env, **args)
    
    args['mode'] = 'drawstuff'
    args['start_frame'] = 1

    args['train'] = False
    
    args['save_period'] = 20


    print("in joystick ground 'env_step_mode'",args['env_step_mode'])
    print("in joystick ground 'experiment_name'",args['experiment_name'])
    if args['train'] == True:
        # load controlvae
        super(SpeedPlayground, playground).try_load(data_file)    
        playground.save_before_train(args)
        playground.train_loop()
    else:
        playground.try_load(data_file)
        playground.run(args['start_frame'])
