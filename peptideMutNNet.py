from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# class peptideMutNNet(nn.Module):
#     def __init__(self, game, args):
#         # game params
#         self.board_x, self.board_y = game.getBoardSize()
#         self.action_size = game.getActionSize()
#         self.args = args

#         super(peptideMutNNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
#         self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

#         self.bn1 = nn.BatchNorm2d(args.num_channels)
#         self.bn2 = nn.BatchNorm2d(args.num_channels)
#         self.bn3 = nn.BatchNorm2d(args.num_channels)
#         self.bn4 = nn.BatchNorm2d(args.num_channels)

#         self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
#         self.fc_bn1 = nn.BatchNorm1d(1024)

#         self.fc2 = nn.Linear(1024, 512)
#         self.fc_bn2 = nn.BatchNorm1d(512)

#         self.fc3 = nn.Linear(512, self.action_size)

#         self.fc4 = nn.Linear(512, 1)

#     def forward(self, s):
#         #                                                           s: batch_size x board_x x board_y
#         s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
#         s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
#         s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
#         s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
#         s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

#         s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
#         s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

#         pi = self.fc3(s)                                                                         # batch_size x action_size
#         v = self.fc4(s)                                                                          # batch_size x 1

#         return F.log_softmax(pi, dim=1), torch.tanh(v)



# import
import os,sys
import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt

import sklearn.metrics as sm

import torch
import torch.nn as nn
import torch.nn.functional as F

rlist = list('ACDEFGHIKLMNPQRSTVWY')

# model
class peptideMutNNet(nn.Module):
  def __init__(self, game, args):
    super().__init__()
    
    self.action_size = game.getActionSize()
    
    self.device = args.device
    self.max_len = args.max_len
    self.dm = args.d_model
    self.nh = args.n_head
    self.df = self.dm * args.d_ffn
    self.nat = args.n_atlayer
    self.dropout = args.dropout
    self.aa_emb = nn.Embedding(21, self.dm)
    self.pe = self.PositionalEncoding()
    self.texn = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                d_model = self.dm,
                nhead = self.nh,
                dim_feedforward = self.df,
                dropout = self.dropout,
                batch_first = True),
                self.nat)
    self.linear_mm = nn.Linear(self.dm, self.dm)
    self.relu = nn.ReLU()
    self.linear_m6 = nn.Linear(self.dm, 6)
    self.linear_m1 = nn.Linear(self.dm, 1) # DEPRECATED but for version 2.3 or earlier
    self.linear_pi = nn.Linear(self.dm, self.action_size) # generate the pi array
    # Initialize the new layer with Xavier initialization and load saved parameters
    self.initialize_linear_pi()
    self.load_model_with_saved_parameters('pretrain_state_dict.0.pkl')
  
  
  def initialize_linear_pi(self):
    nn.init.xavier_uniform_(self.linear_pi.weight)
    if self.linear_pi.bias is not None:
      nn.init.zeros_(self.linear_pi.bias)

  def load_model_with_saved_parameters(self, filepath):
    saved_state_dict = torch.load(filepath)
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in self.state_dict()}
    self.load_state_dict(saved_state_dict, strict=False)
  
  def SaveParameters(self, opfn='model_parameters.pkl'):
    SP_dict = {'d_model':   self.dm, 
               'max_len':   self.max_len, 
               'n_head':    self.nh, 
               'n_atlayer': self.nat, 
               'dropout':   self.dropout, 
               }
    import pickle
    pickle.dump(SP_dict, open(opfn,'wb'))

  def ReportParameters(self):
    print ('Parameters')
    print (' d_model\t%s' %self.dm)
    print (' max_len\t%s' %self.max_len)
    print (' n_head \t%s' %self.nh)
    print (' n_atlayer\t%s' %self.nat)
    print (' dropout\t%s' %self.dropout)
    print (' device \t%s' %self.device)

  def PositionalEncoding(self):
    pe = torch.zeros(self.max_len, self.dm)
    position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
    i2 = torch.arange(0, self.dm, 2, dtype=torch.float)
    div_term = torch.exp( - i2 * np.log(10000) / self.dm )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(self.device)

  def BaseLayer(self, x):
    x = self.aa_emb(x) + self.pe
    # dropout?
    x = self.texn(x)
    # mask?
    return x

  def CLSBaseLayer(self, x):
    x = self.BaseLayer(x)
    x0 = x[:,0]
    x0 = self.linear_mm(x0)
    x0 = self.relu(x0)
    return x0

  # Below is the dG part
  def DGDecompLayer(self, x):
    x = self.CLSBaseLayer(x)
    x = self.linear_m6(x)
    return x.view(-1,2,3)

  def DGLayer(self, x):
    x = self.DGDecompLayer(x)
    x = x.sum(-1)
    x = x[:,1] - x[:,0]
    return x

  def ProbLayer(self, x):
    x = self.DGLayer(x)
    return torch.tanh(x)

  # Below is the pi part
  def PIlayer(self, x):
    x = self.CLSBaseLayer(x)
    x = self.linear_pi(x)
    return F.log_softmax(x, dim=1)
  
# eval
def Metric(model, Xyloader, precision=3):
  model.eval()
  yps, ys = [], []
  for X,y in Xyloader:
    #pred = model.DGLayer(X)
    #yp = torch.sigmoid(pred).cpu().detach().tolist()
    yp = model.ProbLayer(X).cpu().detach().tolist()
    yps = yps + yp
    ys = ys + y.cpu().detach().tolist()
  ys = np.array(ys)
  yps = np.array(yps)
  auc = np.round(sm.roc_auc_score(ys, yps),      precision)
  acc = np.round(sm.accuracy_score(ys, yps>0.5), precision)
  cel = np.round(sm.log_loss(ys, yps),           precision)
  return cel,acc,auc



# mutate
def Mutate(seq, Nmut):
  out = []
  labels = []
  for p in it.combinations(range(len(seq)), Nmut):
    for r in it.product(rlist, repeat=Nmut):
      seq_tmp = list(seq)
      label = ''
      for i in range(len(p)):
        r0i = seq_tmp[p[i]]
        ri = r[i]
        if r0i != ri:
          seq_tmp[p[i]] = ri
          label = label + r0i + str(p[i]+1) + ri + '+'
      out.append(''.join(seq_tmp))
      labels.append(label[:-1])
  return out, labels

def MutatePlot(arr, opfn=None):
  plt.figure(figsize=(6,3))
  plt.imshow(arr, cmap='bwr_r', vmin=-2, vmax=2)
  plt.colorbar(label='Predicted '+r'$\Delta\Delta$'+'G (kcal/mol)', ticks=np.arange(-2,3,1), fraction=0.05, shrink=0.83)
  plt.xlabel('Mutated Residue')
  plt.ylabel('Position')
  plt.xticks(np.arange(20), rlist)
  plt.yticks(np.arange(9), np.arange(1,10))
  if opfn is None:
    plt.show()
  else:
    plt.savefig(opfn)

def MutatePlot_PKNAP(arr, opfn=None):
  from matplotlib import font_manager
  myfont = font_manager.FontProperties
  bg = '#1B1A1A'
  plt.rcParams['text.color']= '#000000'
  plt.figure(figsize=(6,3), facecolor=bg)
  ax = plt.subplot(111)
  ax.set_facecolor(bg)
  plt.imshow(arr, cmap='bwr_r', vmin=-2, vmax=2)
  plt.colorbar(label='Predicted '+r'$\Delta\Delta$'+'G (kcal/mol)', ticks=np.arange(-2,3,1), fraction=0.05, shrink=0.83)
  plt.xlabel('Mutated Residue', color='#000000')
  plt.ylabel('Position')
  plt.xticks(np.arange(20), rlist)
  plt.yticks(np.arange(9), np.arange(1,10))
  if opfn is None:
    plt.show()
  else:
    plt.savefig(opfn)

def ddg_ml(dglayer, otindex):
  #return dglayer[otindex[1]] - dglayer[otindex[0]]
  return dglayer[otindex[0]] - dglayer[otindex[1]]
# DEPRECATED
#def ddgse2_ml(stdlayer, otindex):
#  return stdlayer[otindex[1]]**2 + stdlayer[otindex[0]]**2

def Sigmoid(x):
  return 1/(1+np.exp(-x))

#def AUC(x,y):
#  return sm.roc_auc_score(x>0,Sigmoid(y))

