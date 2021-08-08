from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import agents.sac.core as core
from agents.sac.logx import EpochLogger
from ipdb import set_trace
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch as tc
from parameters import DROPOUT, SAC_WARMUP, SAC_TEST_EVERY, LOCAL, SAC_ALPHA, HUGE, LOAD_MODEL
from torch.distributions.normal import Normal
import os
from auxLib3 import dumpDataStr, loadDataStr


class ReplayBuffer_rw:
    """
    A simple FIFO experience replay buffer for Reward Fn Approximation for SAC agents.
    """

    def __init__(self, obs_dim, ntsk_dim, rw_dim, size):

        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.ntsk_buf = np.zeros(core.combined_shape(size, ntsk_dim), dtype=np.float32)

        self.rew_buf = np.zeros(core.combined_shape(size, rw_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, ntsk, rew):
        self.obs_buf[self.ptr] = obs
        self.rew_buf[self.ptr] = rew
        self.ntsk_buf[self.ptr] = ntsk
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     rew=self.rew_buf[idxs],
                     ntsk=self.ntsk_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_ac_dim, obs_cr_dim, act_dim, size):

        self.obs_ac_buf = np.zeros(core.combined_shape(size, obs_ac_dim), dtype=np.float32)
        self.obs2_ac_buf = np.zeros(core.combined_shape(size, obs_ac_dim), dtype=np.float32)

        self.obs_cr_buf = np.zeros(core.combined_shape(size, obs_cr_dim), dtype=np.float32)
        self.obs2_cr_buf = np.zeros(core.combined_shape(size, obs_cr_dim), dtype=np.float32)

        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)

        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs_ac, obs_cr, act, rew, next_obs_ac, next_obs_cr, done):

        self.obs_ac_buf[self.ptr] = obs_ac
        self.obs2_ac_buf[self.ptr] = next_obs_ac

        self.obs_cr_buf[self.ptr] = obs_cr
        self.obs2_cr_buf[self.ptr] = next_obs_cr

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # self.dr_rew_buf[self.ptr] = dr_rew

        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):

        # set_trace()
        # print("***********")
        # print(self.size, batch_size)

        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs_ac=self.obs_ac_buf[idxs],
                     obs2_ac=self.obs2_ac_buf[idxs],
                     obs_cr=self.obs_cr_buf[idxs],
                     obs2_cr=self.obs2_cr_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     #dr_rew=self.dr_rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class rt_nw(nn.Module):

    def __init__(self, ip_dim, num_edges, op_dim):
        super(rt_nw, self).__init__()

        self.num_edges = num_edges
        self.ip_dim = ip_dim
        self.h_dim1 = 100
        self.h_dim2 = 100
        self.o_dim = op_dim

        self.drop1 = nn.ModuleList()
        self.linear1 = nn.ModuleList()
        self.act1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.drop2 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.act2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()
        self.drop_op = nn.ModuleList()
        self.op = nn.ModuleList()

        for e in range(self.num_edges):
            # ---- Layer1
            self.linear1.append(nn.Linear(ip_dim, self.h_dim1, bias=True))
            self.act1.append(nn.LeakyReLU())
            self.ln1.append(nn.LayerNorm(self.h_dim1))

            # ---- Layer2
            self.drop2.append(nn.Dropout(p=DROPOUT))
            self.linear2.append(nn.Linear(self.h_dim1, self.h_dim2, bias=True))
            self.act2.append(nn.LeakyReLU())
            self.ln2.append(nn.LayerNorm(self.h_dim2))

            # ---- Output
            self.drop_op.append(nn.Dropout(p=DROPOUT))
            self.op.append(nn.Linear(self.h_dim2, self.o_dim, bias=True))

    def forward(self, input):

        dtpt = input.shape[0]
        tmp_op = tc.tensor([])

        for e in range(self.num_edges):

            # Mask
            x = input[:,e,:]
            # Layer1
            # x = self.drop1[e](x)
            x = self.linear1[e](x)
            x = self.act1[e](x)
            x = self.ln1[e](x)

            # Layer2
            x = self.drop2[e](x)
            x = self.linear2[e](x)
            x = self.act2[e](x)
            x = self.ln2[e](x)

            # Output
            x = self.drop_op[e](x)
            op = self.op[e](x)

            # vstack
            tmp_op = tc.cat((tmp_op, op), 0)

        output = tmp_op.reshape(dtpt, self.num_edges, self.o_dim)
        return output

class sac_dr_global:

    def __init__(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1000), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, num_sectors=-1, dir_name=None, pro_folder=None, num_los=-1, eps_dim=-1, max_ac = -1):


        self.writer = SummaryWriter(pro_folder+"/log/"+dir_name+"/plots/")

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.num_sectors = num_sectors
        self.eps_dim = eps_dim
        self.num_los = num_los

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----- Dimensions
        self.obs_ac_dim = self.num_sectors
        self.obs_cr_dim = self.num_sectors + self.num_sectors*self.num_los*self.num_los*self.eps_dim
        self.act_dim = 1
        self.obs_rw_dim = (self.num_sectors, self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los)
        self.ntsk_dim = (self.num_sectors, self.num_los, self.num_los, self.eps_dim)
        self.rw_dim = (self.num_sectors, self.num_los * self.num_los)

        # ---- Rw Nw
        self.replay_buffer_rw = ReplayBuffer_rw(obs_dim=self.obs_rw_dim, ntsk_dim=self.ntsk_dim, rw_dim=self.rw_dim, size=replay_size)

        self.ac_list = []
        self.ac_targ_list = []
        self.q_params_list = []
        self.replay_buffer_list = []
        self.pi_optimizer_list = []
        self.q_optimizer_list = []
        for e in range(self.num_sectors):
            # Create actor-critic module and target networks
            self.ac_list.append(actor_critic(self.obs_ac_dim,self.obs_cr_dim, self.act_dim, **ac_kwargs))
            self.ac_targ_list.append(deepcopy(self.ac_list[e]))

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.ac_targ_list[e].parameters():
                p.requires_grad = False

            # List of parameters for both Q-networks (save this for convenience)
            self.q_params_list.append(itertools.chain(self.ac_list[e].q1.parameters(), self.ac_list[e].q2.parameters()))

            # Experience buffer
            self.replay_buffer_list.append(ReplayBuffer(obs_ac_dim=self.obs_ac_dim,obs_cr_dim=self.obs_cr_dim, act_dim=self.act_dim, size=replay_size))

            # Set up optimizers for policy and q-function
            self.pi_optimizer_list.append(Adam(self.ac_list[e].pi.parameters(), lr=lr))
            self.q_optimizer_list.append(Adam(self.q_params_list[e], lr=lr))

            # Set up model saving
            self.logger.setup_pytorch_saver(self.ac_list[e])


        # ---- Reward Network
        self.num_los = num_los
        self.eps_dim = eps_dim
        self.max_ac = max_ac
        ip_dim = self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los
        op_dim = self.num_los * self.num_los
        self.rt_nw = rt_nw(ip_dim, self.num_sectors, op_dim)
        self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=lr)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%self.var_counts)

    def prepare_obs_cr(self, x, z, z_count):

        x_cr = x.reshape(self.num_sectors*self.num_los*self.num_los*self.eps_dim)
        x_cr = np.array(x_cr.copy())
        x_cr = np.tile(x_cr, (z_count, 1))
        return x_cr

    def compute_diff_rw(self, nt, ntellv, ntsk_indx, df_nts, df_ntsk):

        ntsk = ntellv.reshape(self.num_sectors, self.num_los*self.num_los*self.eps_dim)
        nts = nt.reshape(self.num_sectors, self.num_los*self.num_los)
        nsa = np.concatenate((ntsk, nts), axis=-1)
        input = tc.FloatTensor([nsa])
        input = input / self.max_ac
        input.requires_grad = True

        op1 = self.rt_nw(input)
        op2 = ntellv.sum(3).reshape(1, self.num_sectors, self.num_los*self.num_los)
        op3 = tc.FloatTensor(op2)
        rt_nn = (op1 * op3).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/self.max_ac
        diff_rw = grad.numpy().squeeze()
        diff_rw = np.hsplit(diff_rw, np.array([ntsk_indx,ntsk_indx]))
        dr_ntsk = diff_rw[0]
        dr_nts = diff_rw[2]

        df_ntsk = dr_ntsk[:, df_ntsk].reshape(self.num_sectors, 1)
        df_nts = dr_nts[:, df_nts].reshape(self.num_sectors, 1)

        dr_ntsk = dr_ntsk - df_ntsk
        dr_nts = dr_nts - df_nts

        dr_ntsk = dr_ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr_nts = dr_nts.reshape(self.num_sectors, self.num_los * self.num_los, 1)
        dr = dr_ntsk + dr_nts
        ntsk = ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr = ntsk * dr
        dr_z = dr.sum(-1).sum(-1)

        input.requires_grad = False
        input = input.numpy().squeeze()

        # set_trace()
        # input = input.reshape(self.num_sectors * ( self.num_los*self.num_los*self.eps_dim + self.num_los*self.num_los))
        return dr_z, input

    def compute_loss_q(self, ac, ac_targ, data):
        o_cr, a, r, o2_ac, o2_cr, d = data['obs_cr'], data['act'], data['rew'],data['obs2_ac'], data['obs2_cr'], data['done']

        q1 = ac.q1(o_cr,a)
        q2 = ac.q2(o_cr,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy

            a2, logp_a2, _ = ac.pi(o2_ac)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2_cr, a2)
            q2_pi_targ = ac_targ.q2(o2_cr, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, ac, data):
        o_ac, o_cr = data['obs_ac'], data['obs_cr']
        pi, logp_pi, _ = ac.pi(o_ac)
        q1_pi = ac.q1(o_cr, pi)
        q2_pi = ac.q2(o_cr, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def compute_loss_rw(self, data):

        obs, y_target, ntellv = data['obs'], data['rew'], data['ntsk']


        x = tc.FloatTensor(obs)
        y_pred = self.rt_nw(x)
        y_target = tc.tensor(y_target).float()
        op1 = y_pred - y_target
        dtpt = ntellv.shape[0]
        ntellv = ntellv.reshape(dtpt, self.num_sectors, self.num_los*self.num_los, self.eps_dim)
        op3 = ntellv.sum(-1)
        op4 = op1 * op3
        op4 = op4.sum(-1).sum(-1)
        op5 = op4 * op4
        loss = op5.sum()
        dtpt = dtpt * self.num_sectors
        loss = loss / dtpt
        return loss

    def update_rw(self, batch_size):

        # ---- Train Reward Function
        data = self.replay_buffer_rw.sample_batch(batch_size)
        self.loss_rw = self.compute_loss_rw(data)
        self.rt_nw_opt.zero_grad()
        self.loss_rw.backward()
        self.rt_nw_opt.step()
        self.logger.store(LossR=self.loss_rw.item())

    def update(self, batch_size):

        # Train Actor and Policy Network
        # ---- Pi & Q
        loss_q_list = []
        loss_p_list = []

        for e in range(self.num_sectors):

            data = self.replay_buffer_list[e].sample_batch(batch_size)

            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer_list[e].zero_grad()
            ac = self.ac_list[e]
            ac_targ = self.ac_targ_list[e]
            loss_q, q_info = self.compute_loss_q(ac, ac_targ, data)
            loss_q.backward()
            self.q_optimizer_list[e].step()
            loss_q_list.append(loss_q.item())

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params_list[e]:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer_list[e].zero_grad()
            ac = self.ac_list[e]
            loss_pi, pi_info = self.compute_loss_pi(ac, data)
            loss_pi.backward()
            self.pi_optimizer_list[e].step()
            loss_p_list.append(loss_pi.item())

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params_list[e]:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac_list[e].parameters(), self.ac_targ_list[e].parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        # Record things
        self.logger.store(LossQ=np.mean(loss_q_list), **q_info)
        self.logger.store(LossPi=np.mean(loss_p_list), **pi_info)

    def get_action(self, ac, o, deterministic=False):

        a, noise = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

        return a[0], noise[0]

    def get_random_action(self):

        pi_distribution = Normal(0, 1)
        pi_action = pi_distribution.sample()
        pi_action = torch.tanh(pi_action)

        noise_distribution = Normal(0, 1)
        noise = noise_distribution.sample()

        return pi_action.numpy(), noise.numpy()

    def test_agent(self):

        pass
        # for j in range(self.num_test_episodes):
        #     o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        #     while not(d or (ep_len == max_ep_len)):
        #         # Take deterministic actions at test time
        #         o, r, d, _ = test_env.step(get_action(o, True))
        #         ep_ret += r
        #         ep_len += 1
        #     logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def log(self, ep, ep_rw, avg_tr, avg_cnf, tot_cnf, goal_reached):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)

class sac_dr_indv:

    def __init__(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=-1, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=SAC_ALPHA, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, num_sectors=-1, dir_name=None, pro_folder=None, num_los=-1, eps_dim=-1, max_ac = -1, lg=None):


        self.agent = "sac_dr_indv"
        self.lg = lg
        self.dir_name = dir_name
        self.pro_folder = pro_folder
        self.writer = SummaryWriter(pro_folder+"/log/"+dir_name+"/plots/")

        self.save_flag = False
        self.tcnf_min = HUGE
        self.curr_tcnf_list = []
        self.gamma = gamma
        self.alpha = alpha

        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.num_sectors = num_sectors
        self.eps_dim = eps_dim
        self.num_los = num_los

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.log_flag = False

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----- Dimensions
        self.obs_ac_dim = self.num_sectors
        self.obs_cr_dim = self.num_sectors + self.num_los*self.num_los*self.eps_dim
        self.act_dim = 1
        self.obs_rw_dim = (self.num_sectors, self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los)
        self.ntsk_dim = (self.num_sectors, self.num_los, self.num_los, self.eps_dim)
        self.rw_dim = (self.num_sectors, self.num_los * self.num_los)

        # ---- Rw Nw
        self.replay_buffer_rw = ReplayBuffer_rw(obs_dim=self.obs_rw_dim, ntsk_dim=self.ntsk_dim, rw_dim=self.rw_dim, size=replay_size)

        self.ac_list = []
        self.ac_targ_list = []
        self.q_params_list = []
        self.replay_buffer_list = []
        self.pi_optimizer_list = []
        self.q_optimizer_list = []

        if LOAD_MODEL:
            lg.writeln("-----------------------")
            lg.writeln("Loading Old Model")
            lg.writeln("-----------------------")
            ep = loadDataStr(self.pro_folder + '/load_model/' + self.agent + '_max_ep')
            for e in range(self.num_sectors):
                self.ac_list.append(tc.load(
                    self.pro_folder + '/load_model' + '/model_' + str(e) + "_" + self.agent + "_" + str(ep) + ".pt"))
                self.ac_list[e].eval()
        else:
            for e in range(self.num_sectors):
                # Create actor-critic module and target networks
                self.ac_list.append(actor_critic(self.obs_ac_dim,self.obs_cr_dim, self.act_dim, **ac_kwargs))
                self.ac_targ_list.append(deepcopy(self.ac_list[e]))

                # Freeze target networks with respect to optimizers (only update via polyak averaging)
                for p in self.ac_targ_list[e].parameters():
                    p.requires_grad = False

                # List of parameters for both Q-networks (save this for convenience)
                self.q_params_list.append(itertools.chain(self.ac_list[e].q1.parameters(), self.ac_list[e].q2.parameters()))

                # Experience buffer
                self.replay_buffer_list.append(ReplayBuffer(obs_ac_dim=self.obs_ac_dim,obs_cr_dim=self.obs_cr_dim, act_dim=self.act_dim, size=replay_size))

                # Set up optimizers for policy and q-function
                self.pi_optimizer_list.append(Adam(self.ac_list[e].pi.parameters(), lr=lr))
                self.q_optimizer_list.append(Adam(self.q_params_list[e], lr=lr))


        # ---- Reward Network
        self.num_los = num_los
        self.eps_dim = eps_dim
        self.max_ac = max_ac
        ip_dim = self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los
        op_dim = self.num_los * self.num_los
        if LOAD_MODEL:
            ep = loadDataStr(self.pro_folder + '/load_model/'+self.agent+'_max_ep')
            # Reward Network
            # --------------
            self.rt_nw = tc.load(self.pro_folder + '/load_model' + '/model_rt' + "_" + self.agent +"_"+str(ep) +".pt")
            self.rt_nw.eval()
        else:
            self.rt_nw = rt_nw(ip_dim, self.num_sectors, op_dim)
            self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=lr)


    def prepare_obs_cr(self, x, z, z_count):

        x = x[z]
        x_cr = x.reshape(self.num_los*self.num_los*self.eps_dim)
        x_cr = np.array(x_cr.copy())
        x_cr = np.tile(x_cr, (z_count, 1))
        return x_cr

    def compute_diff_rw(self, nt, ntellv, ntsk_indx, df_nts, df_ntsk):

        ntsk = ntellv.reshape(self.num_sectors, self.num_los*self.num_los*self.eps_dim)
        nts = nt.reshape(self.num_sectors, self.num_los*self.num_los)
        nsa = np.concatenate((ntsk, nts), axis=-1)
        input = tc.FloatTensor([nsa])
        input = input / self.max_ac
        input.requires_grad = True

        op1 = self.rt_nw(input)
        op2 = ntellv.sum(3).reshape(1, self.num_sectors, self.num_los*self.num_los)
        op3 = tc.FloatTensor(op2)
        rt_nn = (op1 * op3).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/self.max_ac
        diff_rw = grad.numpy().squeeze()
        diff_rw = np.hsplit(diff_rw, np.array([ntsk_indx,ntsk_indx]))
        dr_ntsk = diff_rw[0]
        dr_nts = diff_rw[2]

        df_ntsk = dr_ntsk[:, df_ntsk].reshape(self.num_sectors, 1)
        df_nts = dr_nts[:, df_nts].reshape(self.num_sectors, 1)

        dr_ntsk = dr_ntsk - df_ntsk
        dr_nts = dr_nts - df_nts

        dr_ntsk = dr_ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr_nts = dr_nts.reshape(self.num_sectors, self.num_los * self.num_los, 1)
        dr = dr_ntsk + dr_nts
        ntsk = ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr = ntsk * dr
        dr_z = dr.sum(-1).sum(-1)

        input.requires_grad = False
        input = input.numpy().squeeze()

        # set_trace()
        # input = input.reshape(self.num_sectors * ( self.num_los*self.num_los*self.eps_dim + self.num_los*self.num_los))
        return dr_z, input

    def compute_loss_q(self, ac, ac_targ, data):
        o_cr, a, r, o2_ac, o2_cr, d = data['obs_cr'], data['act'], data['rew'],data['obs2_ac'], data['obs2_cr'], data['done']

        q1 = ac.q1(o_cr,a)
        q2 = ac.q2(o_cr,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2, _, _, _ = ac.pi(o2_ac)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2_cr, a2)
            q2_pi_targ = ac_targ.q2(o2_cr, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, ac, data):
        o_ac, o_cr = data['obs_ac'], data['obs_cr']
        pi, logp_pi, _, _, _ = ac.pi(o_ac)
        q1_pi = ac.q1(o_cr, pi)
        q2_pi = ac.q2(o_cr, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def compute_loss_rw(self, data):

        obs, y_target, ntellv = data['obs'], data['rew'], data['ntsk']
        x = tc.FloatTensor(obs)
        y_pred = self.rt_nw(x)
        y_target = tc.tensor(y_target).float()
        op1 = y_pred - y_target
        dtpt = ntellv.shape[0]
        ntellv = ntellv.reshape(dtpt, self.num_sectors, self.num_los*self.num_los, self.eps_dim)
        op3 = ntellv.sum(-1)
        op4 = op1 * op3
        op4 = op4.sum(-1).sum(-1)
        op5 = op4 * op4
        loss = op5.sum()
        dtpt = dtpt * self.num_sectors
        loss = loss / dtpt
        return loss

    def update_rw(self, batch_size):

        # ---- Train Reward Function
        data = self.replay_buffer_rw.sample_batch(batch_size)
        loss_rw = self.compute_loss_rw(data)
        self.rt_nw_opt.zero_grad()
        loss_rw.backward()
        self.rt_nw_opt.step()
        self.loss_rw = loss_rw.item()
        # self.logger.store(LossR=loss_rw.item())

    def update(self, batch_size):

        # Train Actor and Policy Network
        # ---- Pi & Q
        loss_q_list = []
        loss_p_list = []

        for e in range(self.num_sectors):


            data = self.replay_buffer_list[e].sample_batch(batch_size)

            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer_list[e].zero_grad()
            ac = self.ac_list[e]
            ac_targ = self.ac_targ_list[e]
            loss_q, q_info = self.compute_loss_q(ac, ac_targ, data)
            loss_q.backward()
            self.q_optimizer_list[e].step()
            loss_q_list.append(loss_q.item())

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params_list[e]:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer_list[e].zero_grad()
            ac = self.ac_list[e]
            loss_pi, pi_info = self.compute_loss_pi(ac, data)
            loss_pi.backward()
            self.pi_optimizer_list[e].step()
            loss_p_list.append(loss_pi.item())



            # self.writer.add_histogram("Policy/"+str(e)+"_mlp_"+str(0), self.ac_list[e]., ep)
            #
            # self.writer.add_histogram("Policy/"+str(e)+"_mlp_"+str(1), self.ac_list[e].pi.net[2].weight, ep)


            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params_list[e]:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac_list[e].parameters(), self.ac_targ_list[e].parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        # Record things
        self.loss_q = np.mean(loss_q_list)
        self.loss_p = np.mean(loss_p_list)

        # self.log_flag = True
        # self.logger.store(LossQ=np.mean(loss_q_list), **q_info)
        # self.logger.store(LossPi=np.mean(loss_p_list), **pi_info)

    def get_action(self, ac, o, deterministic=False):

        a, epsilon, mu, std = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
        return a[0], epsilon[0], mu[0], std[0]

    def get_random_action(self):

        pi_action = np.random.uniform(0, 1)
        pi_action = np.tanh(pi_action)

        noise_distribution = Normal(0, 1)
        noise = noise_distribution.sample()

        return pi_action, noise.numpy()

    def test_agent(self):

        pass
        # for j in range(self.num_test_episodes):
        #     o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        #     while not(d or (ep_len == max_ep_len)):
        #         # Take deterministic actions at test time
        #         o, r, d, _ = test_env.step(get_action(o, True))
        #         ep_ret += r
        #         ep_len += 1
        #     logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def log(self, ep, train_ep, test_ep, ep_rw, avg_tr, tot_cnf, test_flag, avg_speed, mu_list_train, std_list, mu_list_test):

        # ---- Reward
        if test_flag:
            self.writer.add_scalar('Test/Return', ep_rw, test_ep)
            self.writer.add_scalar('Test/AvgTravelTime', avg_tr, test_ep)
            self.writer.add_scalar('Test/TotalConflicts', tot_cnf, test_ep)
            for e in range(self.num_sectors):
                self.writer.add_scalar('Test/AvgSpeed_'+str(e), np.mean(avg_speed[e]), test_ep)
                self.writer.add_scalar('Test/mu_'+str(e), np.mean(mu_list_test[e]), test_ep)
        else:
            self.writer.add_scalar('Train/Return', ep_rw, train_ep)
            self.writer.add_scalar('Train/AvgTravelTime', avg_tr, train_ep)
            self.writer.add_scalar('Train/TotalConflicts', tot_cnf, train_ep)

            for e in range(self.num_sectors):
                self.writer.add_scalar('Train/AvgSpeed_'+str(e), np.mean(avg_speed[e]), train_ep)
                self.writer.add_scalar('Train/mu_'+str(e), np.mean(mu_list_train[e]), train_ep)
                self.writer.add_scalar('Train/std_'+str(e), np.mean(std_list[e]), train_ep)

        # if LOCAL:
        #     # ---- Metrics
        #     for e in range(self.num_sectors):
        #         # -------- Policy Nw Weights
        #         self.writer.add_histogram("Policy/"+str(e)+"_mlp_"+str(0), self.ac_list[e].pi.net[0].weight, ep)
        #         self.writer.add_histogram("Policy/"+str(e)+"_mlp_"+str(1), self.ac_list[e].pi.net[2].weight, ep)
        #         self.writer.add_histogram("Policy/"+str(e)+"_mu_layer", self.ac_list[e].pi.mu_layer.weight, ep)
        #         self.writer.add_histogram("Policy/"+str(e)+"_log_std_layer", self.ac_list[e].pi.log_std_layer.weight, ep)

    def save_model(self, test_ep):

        # cmd = "rm " + self.pro_folder + '/log/' + self.dir_name + '/model/'
        # os.system(cmd + "*.*")

        dumpDataStr(self.pro_folder + '/log/' + self.dir_name + '/model/' + self.agent + '_max_ep', test_ep)

        # Reward Network
        tc.save(self.rt_nw,
                self.pro_folder + '/log/' + self.dir_name + '/model/model_rt' + "_" + self.agent + "_" + str(test_ep) + ".pt")

        # AC Network
        for e in range(self.num_sectors):
            tc.save(self.ac_list[e], self.pro_folder + '/log/' + self.dir_name + '/model/model_' + str(e) + "_" + self.agent +  "_"+str(test_ep) +".pt")

class sac_random:

    def __init__(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=-1, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=SAC_ALPHA, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, num_sectors=-1, dir_name=None, pro_folder=None, num_los=-1, eps_dim=-1, max_ac = -1):

        self.writer = SummaryWriter(pro_folder+"/log/"+dir_name+"/plots/")

        self.gamma = gamma
        self.alpha = alpha

        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.num_sectors = num_sectors
        self.eps_dim = eps_dim
        self.num_los = num_los

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----- Dimensions
        self.obs_ac_dim = self.num_sectors
        self.obs_cr_dim = self.num_sectors + self.num_los*self.num_los*self.eps_dim
        self.act_dim = 1
        self.obs_rw_dim = (self.num_sectors, self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los)
        self.ntsk_dim = (self.num_sectors, self.num_los, self.num_los, self.eps_dim)
        self.rw_dim = (self.num_sectors, self.num_los * self.num_los)

        # ---- Rw Nw
        self.replay_buffer_rw = ReplayBuffer_rw(obs_dim=self.obs_rw_dim, ntsk_dim=self.ntsk_dim, rw_dim=self.rw_dim, size=replay_size)

        self.ac_list = []
        self.ac_targ_list = []
        self.q_params_list = []
        self.replay_buffer_list = []
        self.pi_optimizer_list = []
        self.q_optimizer_list = []
        for e in range(self.num_sectors):
            # Create actor-critic module and target networks
            self.ac_list.append(actor_critic(self.obs_ac_dim,self.obs_cr_dim, self.act_dim, **ac_kwargs))
            self.ac_targ_list.append(deepcopy(self.ac_list[e]))

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.ac_targ_list[e].parameters():
                p.requires_grad = False

            # List of parameters for both Q-networks (save this for convenience)
            self.q_params_list.append(itertools.chain(self.ac_list[e].q1.parameters(), self.ac_list[e].q2.parameters()))

            # Experience buffer
            self.replay_buffer_list.append(ReplayBuffer(obs_ac_dim=self.obs_ac_dim,obs_cr_dim=self.obs_cr_dim, act_dim=self.act_dim, size=replay_size))

            # Set up optimizers for policy and q-function
            self.pi_optimizer_list.append(Adam(self.ac_list[e].pi.parameters(), lr=lr))
            self.q_optimizer_list.append(Adam(self.q_params_list[e], lr=lr))

            # Set up model saving
            self.logger.setup_pytorch_saver(self.ac_list[e])


        # ---- Reward Network
        self.num_los = num_los
        self.eps_dim = eps_dim
        self.max_ac = max_ac
        ip_dim = self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los
        op_dim = self.num_los * self.num_los
        self.rt_nw = rt_nw(ip_dim, self.num_sectors, op_dim)
        self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=lr)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%self.var_counts)

    def prepare_obs_cr(self, x, z, z_count):

        x = x[z]
        x_cr = x.reshape(self.num_los*self.num_los*self.eps_dim)
        x_cr = np.array(x_cr.copy())
        x_cr = np.tile(x_cr, (z_count, 1))
        return x_cr

    def compute_diff_rw(self, nt, ntellv, ntsk_indx, df_nts, df_ntsk):

        ntsk = ntellv.reshape(self.num_sectors, self.num_los*self.num_los*self.eps_dim)
        nts = nt.reshape(self.num_sectors, self.num_los*self.num_los)
        nsa = np.concatenate((ntsk, nts), axis=-1)
        input = tc.FloatTensor([nsa])
        input = input / self.max_ac
        input.requires_grad = True

        op1 = self.rt_nw(input)
        op2 = ntellv.sum(3).reshape(1, self.num_sectors, self.num_los*self.num_los)
        op3 = tc.FloatTensor(op2)
        rt_nn = (op1 * op3).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/self.max_ac
        diff_rw = grad.numpy().squeeze()
        diff_rw = np.hsplit(diff_rw, np.array([ntsk_indx,ntsk_indx]))
        dr_ntsk = diff_rw[0]
        dr_nts = diff_rw[2]

        df_ntsk = dr_ntsk[:, df_ntsk].reshape(self.num_sectors, 1)
        df_nts = dr_nts[:, df_nts].reshape(self.num_sectors, 1)

        dr_ntsk = dr_ntsk - df_ntsk
        dr_nts = dr_nts - df_nts

        dr_ntsk = dr_ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr_nts = dr_nts.reshape(self.num_sectors, self.num_los * self.num_los, 1)
        dr = dr_ntsk + dr_nts
        ntsk = ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr = ntsk * dr
        dr_z = dr.sum(-1).sum(-1)

        input.requires_grad = False
        input = input.numpy().squeeze()

        # set_trace()
        # input = input.reshape(self.num_sectors * ( self.num_los*self.num_los*self.eps_dim + self.num_los*self.num_los))
        return dr_z, input

    def compute_loss_q(self, ac, ac_targ, data):
        o_cr, a, r, o2_ac, o2_cr, d = data['obs_cr'], data['act'], data['rew'],data['obs2_ac'], data['obs2_cr'], data['done']

        q1 = ac.q1(o_cr,a)
        q2 = ac.q2(o_cr,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy

            a2, logp_a2, _ = ac.pi(o2_ac)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2_cr, a2)
            q2_pi_targ = ac_targ.q2(o2_cr, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, ac, data):
        o_ac, o_cr = data['obs_ac'], data['obs_cr']
        pi, logp_pi, _ = ac.pi(o_ac)
        q1_pi = ac.q1(o_cr, pi)
        q2_pi = ac.q2(o_cr, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()



        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def compute_loss_rw(self, data):

        obs, y_target, ntellv = data['obs'], data['rew'], data['ntsk']
        x = tc.FloatTensor(obs)
        y_pred = self.rt_nw(x)
        y_target = tc.tensor(y_target).float()
        op1 = y_pred - y_target
        dtpt = ntellv.shape[0]
        ntellv = ntellv.reshape(dtpt, self.num_sectors, self.num_los*self.num_los, self.eps_dim)
        op3 = ntellv.sum(-1)
        op4 = op1 * op3
        op4 = op4.sum(-1).sum(-1)
        op5 = op4 * op4
        loss = op5.sum()
        dtpt = dtpt * self.num_sectors
        loss = loss / dtpt
        return loss

    def update_rw(self, batch_size):

        self.logger.store(LossR=-1)

    def update(self, batch_size):

        self.logger.store(LossQ=-1)
        self.logger.store(LossPi=-1)

    def get_action(self, ac, o, deterministic=False):

        a = np.random.uniform(-1, 1, size=1)
        noise = np.random.standard_normal(size=1)
        return a[0], noise[0]

    def get_random_action(self):

        pi_distribution = Normal(0, 1)
        pi_action = pi_distribution.sample()
        pi_action = torch.tanh(pi_action)

        noise_distribution = Normal(0, 1)
        noise = noise_distribution.sample()

        return pi_action.numpy(), noise.numpy()

    def test_agent(self):

        pass
        # for j in range(self.num_test_episodes):
        #     o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        #     while not(d or (ep_len == max_ep_len)):
        #         # Take deterministic actions at test time
        #         o, r, d, _ = test_env.step(get_action(o, True))
        #         ep_ret += r
        #         ep_len += 1
        #     logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def log(self, ep, train_ep, test_ep, ep_rw, avg_tr, tot_cnf, test_flag, avg_speed):

        # ---- Reward
        if test_flag:
            self.writer.add_scalar('Test/Return', ep_rw, test_ep)
            self.writer.add_scalar('Test/AvgTravelTime', avg_tr, test_ep)
            self.writer.add_scalar('Test/TotalConflicts', tot_cnf, test_ep)
            for e in range(self.num_sectors):
                self.writer.add_scalar('Test/AvgSpeed_'+str(e), np.mean(avg_speed[e]), test_ep)

        else:
            self.writer.add_scalar('Train/Return', ep_rw, train_ep)
            self.writer.add_scalar('Train/AvgTravelTime', avg_tr, train_ep)
            self.writer.add_scalar('Train/TotalConflicts', tot_cnf, train_ep)


            for e in range(self.num_sectors):
                self.writer.add_scalar('Train/AvgSpeed_'+str(e), np.mean(avg_speed[e]), train_ep)



        if LOCAL:
            # ---- Metrics
            for e in range(self.num_sectors):

                # -------- Policy Nw Weights
                self.writer.add_histogram("Policy/"+str(e)+"_mlp_"+str(0), self.ac_list[e].pi.net[0].weight, ep)

                self.writer.add_histogram("Policy/"+str(e)+"_mlp_"+str(1), self.ac_list[e].pi.net[2].weight, ep)

                self.writer.add_histogram("Policy/"+str(e)+"_mu_layer", self.ac_list[e].pi.mu_layer.weight, ep)

                self.writer.add_histogram("Policy/"+str(e)+"_log_std_layer", self.ac_list[e].pi.log_std_layer.weight, ep)

class sac_rw_indv:

    def __init__(self, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=-1, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1, num_sectors=-1, dir_name=None, pro_folder=None, num_los=-1, eps_dim=-1, max_ac = -1):


        self.writer = SummaryWriter(pro_folder+"/log/"+dir_name+"/plots/")

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.num_sectors = num_sectors
        self.eps_dim = eps_dim
        self.num_los = num_los

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        torch.manual_seed(seed)
        np.random.seed(seed)

        # ----- Dimensions
        self.obs_ac_dim = self.num_sectors
        self.obs_cr_dim = self.num_sectors + self.num_los*self.num_los*self.eps_dim
        self.act_dim = 1
        self.obs_rw_dim = (self.num_sectors, self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los)
        self.ntsk_dim = (self.num_sectors, self.num_los, self.num_los, self.eps_dim)
        self.rw_dim = (self.num_sectors, self.num_los * self.num_los)

        # ---- Rw Nw
        self.replay_buffer_rw = ReplayBuffer_rw(obs_dim=self.obs_rw_dim, ntsk_dim=self.ntsk_dim, rw_dim=self.rw_dim, size=replay_size)

        self.ac_list = []
        self.ac_targ_list = []
        self.q_params_list = []
        self.replay_buffer_list = []
        self.pi_optimizer_list = []
        self.q_optimizer_list = []
        for e in range(self.num_sectors):
            # Create actor-critic module and target networks
            self.ac_list.append(actor_critic(self.obs_ac_dim,self.obs_cr_dim, self.act_dim, **ac_kwargs))
            self.ac_targ_list.append(deepcopy(self.ac_list[e]))

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.ac_targ_list[e].parameters():
                p.requires_grad = False

            # List of parameters for both Q-networks (save this for convenience)
            self.q_params_list.append(itertools.chain(self.ac_list[e].q1.parameters(), self.ac_list[e].q2.parameters()))

            # Experience buffer
            self.replay_buffer_list.append(ReplayBuffer(obs_ac_dim=self.obs_ac_dim,obs_cr_dim=self.obs_cr_dim, act_dim=self.act_dim, size=replay_size))

            # Set up optimizers for policy and q-function
            self.pi_optimizer_list.append(Adam(self.ac_list[e].pi.parameters(), lr=lr))
            self.q_optimizer_list.append(Adam(self.q_params_list[e], lr=lr))

            # Set up model saving
            self.logger.setup_pytorch_saver(self.ac_list[e])


        # ---- Reward Network
        self.num_los = num_los
        self.eps_dim = eps_dim
        self.max_ac = max_ac
        ip_dim = self.num_los * self.num_los * self.eps_dim + self.num_los * self.num_los
        op_dim = self.num_los * self.num_los
        self.rt_nw = rt_nw(ip_dim, self.num_sectors, op_dim)
        self.rt_nw_opt = tc.optim.Adam(self.rt_nw.parameters(), lr=lr)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        # self.var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%self.var_counts)

    def prepare_obs_cr(self, x, z, z_count):

        x = x[z]
        x_cr = x.reshape(self.num_los*self.num_los*self.eps_dim)
        x_cr = np.array(x_cr.copy())
        x_cr = np.tile(x_cr, (z_count, 1))
        return x_cr

    def compute_diff_rw(self, nt, ntellv, ntsk_indx, df_nts, df_ntsk):

        ntsk = ntellv.reshape(self.num_sectors, self.num_los*self.num_los*self.eps_dim)
        nts = nt.reshape(self.num_sectors, self.num_los*self.num_los)
        nsa = np.concatenate((ntsk, nts), axis=-1)
        input = tc.FloatTensor([nsa])
        input = input / self.max_ac
        input.requires_grad = True

        op1 = self.rt_nw(input)
        op2 = ntellv.sum(3).reshape(1, self.num_sectors, self.num_los*self.num_los)
        op3 = tc.FloatTensor(op2)
        rt_nn = (op1 * op3).sum()
        grad = tc.autograd.grad(rt_nn, input)[0]
        grad = grad/self.max_ac
        diff_rw = grad.numpy().squeeze()
        diff_rw = np.hsplit(diff_rw, np.array([ntsk_indx,ntsk_indx]))
        dr_ntsk = diff_rw[0]
        dr_nts = diff_rw[2]

        df_ntsk = dr_ntsk[:, df_ntsk].reshape(self.num_sectors, 1)
        df_nts = dr_nts[:, df_nts].reshape(self.num_sectors, 1)

        dr_ntsk = dr_ntsk - df_ntsk
        dr_nts = dr_nts - df_nts

        dr_ntsk = dr_ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr_nts = dr_nts.reshape(self.num_sectors, self.num_los * self.num_los, 1)
        dr = dr_ntsk + dr_nts
        ntsk = ntsk.reshape(self.num_sectors, self.num_los * self.num_los, self.eps_dim)
        dr = ntsk * dr
        dr_z = dr.sum(-1).sum(-1)

        input.requires_grad = False
        input = input.numpy().squeeze()

        # set_trace()
        # input = input.reshape(self.num_sectors * ( self.num_los*self.num_los*self.eps_dim + self.num_los*self.num_los))
        return dr_z, input

    def compute_loss_q(self, ac, ac_targ, data):
        o_cr, a, r, o2_ac, o2_cr, d = data['obs_cr'], data['act'], data['rew'],data['obs2_ac'], data['obs2_cr'], data['done']

        q1 = ac.q1(o_cr,a)
        q2 = ac.q2(o_cr,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy

            a2, logp_a2, _ = ac.pi(o2_ac)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2_cr, a2)
            q2_pi_targ = ac_targ.q2(o2_cr, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, ac, data):
        o_ac, o_cr = data['obs_ac'], data['obs_cr']
        pi, logp_pi, _ = ac.pi(o_ac)
        q1_pi = ac.q1(o_cr, pi)
        q2_pi = ac.q2(o_cr, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def compute_loss_rw(self, data):

        obs, y_target, ntellv = data['obs'], data['rew'], data['ntsk']

        x = tc.FloatTensor(obs)
        y_pred = self.rt_nw(x)
        y_target = tc.tensor(y_target).float()
        op1 = y_pred - y_target
        dtpt = ntellv.shape[0]
        ntellv = ntellv.reshape(dtpt, self.num_sectors, self.num_los*self.num_los, self.eps_dim)
        op3 = ntellv.sum(-1)
        op4 = op1 * op3
        op4 = op4.sum(-1).sum(-1)
        op5 = op4 * op4
        loss = op5.sum()
        dtpt = dtpt * self.num_sectors
        loss = loss / dtpt
        return loss

    def update_rw(self, batch_size):


        # ---- Train Reward Function
        data = self.replay_buffer_rw.sample_batch(batch_size)
        loss_rw = self.compute_loss_rw(data)
        self.rt_nw_opt.zero_grad()
        loss_rw.backward()
        self.rt_nw_opt.step()
        self.logger.store(LossR=loss_rw.item())

    def update(self, batch_size):

        # Train Actor and Policy Network
        # ---- Pi & Q
        loss_q_list = []
        loss_p_list = []

        for e in range(self.num_sectors):


            data = self.replay_buffer_list[e].sample_batch(batch_size)

            # First run one gradient descent step for Q1 and Q2
            self.q_optimizer_list[e].zero_grad()
            ac = self.ac_list[e]
            ac_targ = self.ac_targ_list[e]
            loss_q, q_info = self.compute_loss_q(ac, ac_targ, data)
            loss_q.backward()
            self.q_optimizer_list[e].step()
            loss_q_list.append(loss_q.item())

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in self.q_params_list[e]:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            self.pi_optimizer_list[e].zero_grad()
            ac = self.ac_list[e]
            loss_pi, pi_info = self.compute_loss_pi(ac, data)
            loss_pi.backward()
            self.pi_optimizer_list[e].step()
            loss_p_list.append(loss_pi.item())

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in self.q_params_list[e]:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.ac_list[e].parameters(), self.ac_targ_list[e].parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

        # Record things
        self.logger.store(LossQ=np.mean(loss_q_list), **q_info)
        self.logger.store(LossPi=np.mean(loss_p_list), **pi_info)

    def get_action(self, ac, o, deterministic=False):

        a, noise = ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

        return a[0], noise[0]

    def get_random_action(self):

        pi_distribution = Normal(0, 1)
        pi_action = pi_distribution.sample()
        pi_action = torch.tanh(pi_action)

        noise_distribution = Normal(0, 1)
        noise = noise_distribution.sample()

        return pi_action.numpy(), noise.numpy()

    def test_agent(self):

        pass
        # for j in range(self.num_test_episodes):
        #     o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        #     while not(d or (ep_len == max_ep_len)):
        #         # Take deterministic actions at test time
        #         o, r, d, _ = test_env.step(get_action(o, True))
        #         ep_ret += r
        #         ep_len += 1
        #     logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def log(self, ep, ep_rw, avg_tr, avg_cnf, tot_cnf, goal_reached):

        # ---- Reward
        self.writer.add_scalar('Reward/Total Rewards', ep_rw, ep)

        # ---- Metrics
        self.writer.add_scalar('Metrics/AvgTravelTime', avg_tr, ep)
        self.writer.add_scalar('Metrics/AvgConflicts', avg_cnf, ep)
        self.writer.add_scalar('Metrics/TotalConflicts', tot_cnf, ep)
        self.writer.add_scalar('Metrics/GoalReached', goal_reached, ep)


'''
def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from agents.sac.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
