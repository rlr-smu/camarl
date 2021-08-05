"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 27 Apr 2019
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys

sys.dont_write_bytecode = True
import os
from pprint import pprint
import time
import auxLib as ax
import pdb
import rlcompleter
import time
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from parameters import SEED, NUM_CORES, NUM_OPTIONS, LEARNING_RATE, BATCH_SIZE, HORIZON, DISCOUNT, EPS_START, EPS_DECAY, EPS_END, TINY, DROPOUT, GRADIENT_CLIP, TOTAL_VESSEL, WEIGHT_VARIANCE, LOAD_MODEL, ENTROPY_WEIGHT
from dashBrd import lgFile
import networkx as nx
from tensorboardX import SummaryWriter
# pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
# pdb.set_trace()

tc.manual_seed(SEED)
tc.set_num_threads(NUM_CORES)

# =============================== Variables ================================== #


# ============================================================================ #

class network(nn.Module):

    def __init__(self, totalZONES):

        super(network, self).__init__()

        self.totalZONES = totalZONES
        self.iDim = 2  # < nt(z), nt(z')>
        self.hDim_1 = 2 * self.iDim
        self.hDim_2 = 2 * self.iDim

        self.linear1 = nn.ModuleList()
        self.tanh1 = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.tanh2 = nn.ModuleList()
        self.ln2 = nn.ModuleList()

        self.beta = nn.ModuleList()
        self.sigmoid = nn.ModuleList()

        self.pi = nn.ModuleList()
        self.softmax = nn.ModuleList()

        layerIdx = 0
        for z in range(self.totalZONES):
            for zp in range(self.totalZONES):
                self.linear1.append(nn.Linear(self.iDim, self.hDim_1, bias=True))
                self.linear1[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.linear1[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.tanh1.append(nn.Tanh())
                self.ln1.append(nn.LayerNorm(self.hDim_1))

                self.linear2.append(nn.Linear(self.hDim_1, self.hDim_2, bias=True))
                self.linear2[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.linear2[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.tanh2.append(nn.Tanh())
                self.ln2.append(nn.LayerNorm(self.hDim_2))

                self.pi.append(nn.Linear(self.hDim_2, NUM_OPTIONS, bias=True))
                self.pi[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.pi[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.softmax.append(nn.Softmax(dim=-1))

                self.beta.append(nn.Linear(self.hDim_2, NUM_OPTIONS, bias=True))
                self.beta[layerIdx].weight.data.normal_(0, WEIGHT_VARIANCE)
                self.beta[layerIdx].bias.data.normal_(0, WEIGHT_VARIANCE)
                self.sigmoid.append(nn.Sigmoid())

                layerIdx += 1

    def forward(self, nt):

        nt = tc.tensor(nt).float()
        dtPt = nt.shape[0]
        layerIdx = 0
        output = tc.tensor([])
        output_lg_pi = tc.tensor([])
        output_beta = tc.tensor([])


        for z in range(self.totalZONES):
            local_output = []
            local_output_log_pi = []
            local_output_beta = []

            for zp in range(self.totalZONES):
                x = []
                x.append(nt[:, z])
                x.append(nt[:, zp])

                # 1st Layer
                x = tc.stack(x, 1)
                x = self.linear1[layerIdx](x)
                x = self.tanh1[layerIdx](x)
                x = self.ln1[layerIdx](x)

                # 2nd Layer
                x = self.linear2[layerIdx](x)
                x = self.tanh2[layerIdx](x)
                x = self.ln2[layerIdx](x)

                # Policy
                pi_x = self.pi[layerIdx](x)
                pi = self.softmax[layerIdx](pi_x)
                log_pi = tc.log(pi)
                local_output.append(pi)
                local_output_log_pi.append(log_pi)

                # beta
                beta_x = self.beta[layerIdx](x)
                b = self.sigmoid[layerIdx](beta_x)
                local_output_beta.append(b)

                layerIdx += 1

            local_output = tc.stack(local_output, 1)
            output = tc.cat((output, local_output), 1)
            local_output_log_pi = tc.stack(local_output_log_pi, 1)
            output_lg_pi = tc.cat((output_lg_pi, local_output_log_pi), 1)

            local_output_beta = tc.stack(local_output_beta, 1)
            output_beta = tc.cat((output_beta, local_output_beta), 1)


        output = tc.reshape(output, (dtPt, self.totalZONES, self.totalZONES, NUM_OPTIONS))
        output_lg_pi = tc.reshape(output_lg_pi, (dtPt, self.totalZONES, self.totalZONES, NUM_OPTIONS))
        output_beta = tc.reshape(output_beta, (dtPt, self.totalZONES, self.totalZONES, NUM_OPTIONS))

        return {'pi': output, 'log_pi': output_lg_pi, 'xi': output_beta}

class op_pgv:

    def __init__(self, data=None,  metrics=[], dirName="", load_path=""):

        self.totalZONES = data.totalZONES
        self.planningZONES = data.planningZONES
        self.T_min_max = data.T_min_max
        # self.arrivalTime = data.arrivalTime
        self.totalVESSELS = data.total_vessels
        self.zGraph = data.zGraph
        self.network = network(self.totalZONES)
        if LOAD_MODEL:
            print ("-----------------------")
            print ("Loading Old Model")
            print ("-----------------------")
            # self.network = tc.load("./log"+"/"+dirName+'/model/model.pt')
            self.network = tc.load("./loadModel" + "/" + load_path + '/model/model.pt')
            self.network.eval()

        self.optimizer = tc.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        self.steps_done = 0
        self.clearBuffer()
        self.training_param()
        self.dirName = dirName
        self.lg = lgFile(param=metrics, path="log/" + self.dirName + "/plots/")
        self.q_loss = 0
        self.total_loss = 0
        self.beta_loss = 0
        self.writer = SummaryWriter()

    def training_param(self):

        self.Adv_option = []
        self.Adv_term = []
        self.nt_z_o_tau_batch = []
        self.nt_z_o_batch = []

        self.dataPt = BATCH_SIZE * HORIZON
        # tau' - t - Tmin_zz
        self.tau_p_t_tmin = np.zeros((self.dataPt, self.totalZONES, self.totalZONES, NUM_OPTIONS, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(self.totalZONES):
                    for zp in range(self.totalZONES):
                        tMin_zz = self.T_min_max[z][zp][0]
                        tMax_zz = self.T_min_max[z][zp][1]
                        for o in range(NUM_OPTIONS):
                            for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                                if tau_p < HORIZON:
                                    self.tau_p_t_tmin[tmpDataPt][z][zp][o][tau_p] = tau_p - t - tMin_zz
                tmpDataPt += 1
        self.tau_p_t_tmin = tc.tensor(self.tau_p_t_tmin)
        # Tmax_zz - tau' + t
        self.tmax_taup_t = np.zeros((self.dataPt, self.totalZONES, self.totalZONES, NUM_OPTIONS, HORIZON + 1))
        tmpDataPt = 0
        for b in range(BATCH_SIZE):
            for t in range(HORIZON):
                for z in range(self.totalZONES):
                    for zp in range(self.totalZONES):
                        tMin_zz = self.T_min_max[z][zp][0]
                        tMax_zz = self.T_min_max[z][zp][1]
                        for o in range(NUM_OPTIONS):
                            for tau_p in range(t + tMin_zz, t + tMax_zz + 1):
                                if tau_p < HORIZON:
                                    self.tmax_taup_t[tmpDataPt][z][zp][o][tau_p] = tMax_zz - tau_p + t
                tmpDataPt += 1
        self.tmax_taup_t = tc.tensor(self.tmax_taup_t)

        self.epoch = 1

    def getOption(self, cT):

        with tc.no_grad():
            nt_z = cT.nt_z
            x = np.reshape(nt_z, (1, self.totalZONES))
            nw = self.network(x)
            pi = nw['pi']
            dist = Categorical(pi)
            op = dist.sample().squeeze(0).unsqueeze(-1)
            xi_all = nw['xi'].squeeze(0)
            return op.squeeze(-1).numpy(), pi.squeeze(0).numpy(), xi_all.numpy()

    def clearBuffer(self):

        self.Return_batch = []
        self.nt_z_o_batch = []
        self.nt_z_o_tau_batch = []
        self.nt_batch = []
        self.Adv_option = []
        self.Adv_term = []

    # def storeRollouts(self, buffer_nt_z, buffer_option, buffer_rt, buffer_ntz_zt, buffer_nt_z_o, buffer_rt_z, buffer_nt_z_o_tau, buffer_nt_ztz):

    def storeRollouts(self, buffer_nt_z=None, buffer_nt_z_o=None, buffer_rt=None, buffer_rt_z=None,
                          buffer_nt_z_o_tau=None):

        # ------- Return
        return_so_far = 0
        tmpReturn = []
        for t in range(len(buffer_rt)-1, -1, -1):
            return_so_far = buffer_rt[t] + DISCOUNT * return_so_far
            tmpReturn.append(return_so_far)
        tmpReturn = tmpReturn[::-1]

        # -------- Store
        self.Return_batch.extend(tmpReturn)
        self.nt_batch.extend(buffer_nt_z)
        self.nt_z_o_batch.extend(buffer_nt_z_o)
        self.nt_z_o_tau_batch.extend(buffer_nt_z_o_tau)

        # ------- Decision Option Count
        # buffer_nt_z_o = buffer_nt_z_o/self.totalVESSELS
        # buffer_ntz_zt = buffer_ntz_zt / self.totalVESSELS
        # buffer_nt_z_o_tau = buffer_nt_z_o_tau/self.totalVESSELS

        # ------ Advantage_return
        adv_term = np.zeros((HORIZON, self.totalZONES, self.totalZONES, NUM_OPTIONS, HORIZON + 1))
        adv_option = np.zeros((HORIZON, self.totalZONES, self.totalZONES, NUM_OPTIONS))
        for t in range(HORIZON):
            for z in self.planningZONES:
                for z_p in nx.neighbors(self.zGraph, z):
                    tmin = self.T_min_max[z][z_p][0]
                    tmax = self.T_min_max[z][z_p][1]
                    for o in range(NUM_OPTIONS):
                        adv_option[t][z][z_p][o] = tmpReturn[t]
                        for tau in range(t + tmin, min(t + tmax + 1, HORIZON + 1)):
                            adv_term[t][z][z_p][o][tau] = tmpReturn[t]

        self.Adv_option.extend(adv_option)
        self.Adv_term.extend(adv_term)

        # pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
        # pdb.set_trace()

    def train(self):

        dtPt = BATCH_SIZE * HORIZON

        adv_term = tc.tensor(self.Adv_term).float()
        adv_opt = tc.tensor(self.Adv_option).float()
        adv_term = adv_term.detach()
        adv_opt = adv_opt.detach()

        nw = self.network(tc.tensor(self.nt_batch))

        # -------------------------- Termination
        xi = nw['xi']
        nt_z_o_tau = tc.tensor(self.nt_z_o_tau_batch)

        # ---- log(xi^t_zz)
        xi_log =  tc.log(tc.add(xi, TINY)).double()
        xi_log = tc.reshape(xi_log, (dtPt, self.totalZONES, self.totalZONES, NUM_OPTIONS, 1))

        # ---- log(1 - xi^t_zz)
        ones = tc.ones(dtPt, self.totalZONES, self.totalZONES, NUM_OPTIONS)
        one_xi = tc.sub(ones, xi)
        one_xi_log = tc.log(tc.add(one_xi, TINY)).double()
        one_xi_log = tc.reshape(one_xi_log, (dtPt, self.totalZONES, self.totalZONES, NUM_OPTIONS, 1))

        # ----- nt(z, z', o, tau_p) * [(tau' - t - tmin_zz)*log(beta_t_zz) + (tmax_zz - (tau' - t)) * (1 - log(beta_t_zz))]
        op3 = tc.mul(self.tau_p_t_tmin, xi_log)
        op4 = tc.mul(self.tmax_taup_t, one_xi_log)
        op5 = tc.add(op3, op4)
        op6 = tc.mul(op5, nt_z_o_tau)
        op7 = tc.mul(op6, adv_term.double())
        op7 = tc.reshape(op7, (BATCH_SIZE * HORIZON, self.totalZONES, self.totalZONES, NUM_OPTIONS * (HORIZON + 1)))

        op8 = tc.sum(op7, 3)
        op8 = op8.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        op8 = op8.transpose_(0, 1)
        op8 = op8.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)
        beta_loss = tc.sum(-tc.mean(op8, 2))


        # -------------- Actor
        # ---- Entropy
        pi = nw['pi']
        dist = Categorical(pi)
        ent = dist.entropy()
        ent = ent.reshape(BATCH_SIZE*HORIZON, self.totalZONES*self.totalZONES)
        ent = ent.transpose_(0, 1)
        ent = ent.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)
        ent = tc.mul(ent, ENTROPY_WEIGHT)
        log_pi = nw['log_pi']
        nt_z_zp_o = tc.tensor(self.nt_z_o_batch)
        op1 = tc.mul(log_pi, nt_z_zp_o.float())
        op2 = tc.mul(op1, adv_opt)
        op2 = tc.sum(op2, 3)
        op2 = op2.reshape(BATCH_SIZE * HORIZON, self.totalZONES * self.totalZONES)
        op2 = op2.transpose_(0, 1)
        op2 = op2.reshape(self.totalZONES, self.totalZONES, BATCH_SIZE * HORIZON)
        op3 = tc.add(op2, ent)
        pi_loss = tc.sum(-tc.mean(op3, 2)).double()

        # -------------- Total Loss
        self.optimizer.zero_grad()
        loss = beta_loss+pi_loss
        loss.backward()
        self.optimizer.step()
        self.total_loss = float(beta_loss+pi_loss.data)

        # Clear
        self.clearBuffer()


    def log(self, i, epRw, rw, vio, delay, op, xi_avg2, xi_all_avg2, ent_avg2):

        self.lg.update({'x': i, 'reward': rw, 'vio': vio, 'delay': delay, 'q_loss':self.q_loss, 'beta_loss':self.beta_loss, "total_loss":self.total_loss})

        # self.writer.add_scalar('Sample Avg. Rewards', rw, i)
        self.writer.add_scalar('Total Rewards', epRw, i)
        self.writer.add_scalar('Total ResVio', vio, i)
        self.writer.add_scalar('Total Delay', delay, i)

        for z in self.planningZONES:
            zp_nbr = nx.neighbors(self.zGraph, z)
            for zp in zp_nbr:
                self.writer.add_scalar("xi_values/" + str(z) + "_" + str(zp),
                                       xi_avg2[(z, zp)], i)
                self.writer.add_scalar("Entropy/" + str(z) + "_" + str(zp),
                                       ent_avg2[(z, zp)], i)

                for o in range(NUM_OPTIONS):
                    self.writer.add_scalar("Options_Distribution/"+str(z)+"_"+str(zp)+"_"+str(o), op[(z, zp)][o], i)
                    self.writer.add_scalar("Options_Values/"+str(z)+"_"+str(zp)+"_"+str(o), xi_all_avg2[(z, zp)][o], i)

    def save_model(self):

        tc.save(self.network, 'log/'+self.dirName+'/model/model.pt')

def main():
    print ("Hello World")


# =============================================================================== #

if __name__ == '__main__':
    main()
    