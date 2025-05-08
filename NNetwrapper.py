import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
import torch.nn as nn

from peptideMutNNet import peptideMutNNet as pmnet

args = dotdict({
    'lr': 0.001,
    'weight_decay':0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    
    'num_channels': 512,
    'd_model': 32,
    'max_len':10, 
    'n_head':4, 
    'n_atlayer':1, 
    'd_ffn':4, 
    'device': 'cuda' if torch.cuda.is_available()else 'cpu',
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = pmnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        
        if args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # Freeze the parameters
        for name, param in self.nnet.named_parameters():
            param.requires_grad = False
            if 'linear_m6' in name or 'linear_mm' in name or 'linear_pi' in name:
                param.requires_grad = True
        
        #optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)          
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.nnet.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                sequences, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                seq2nums = torch.tensor(Embedding_CLS(sequences))
                
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    seq2nums, target_pis, target_vs = seq2nums.cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_v = self.nnet.ProbLayer(seq2nums)
                out_pi = self.nnet.PIlayer(seq2nums)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # print("seq2nums number is:", seq2nums.size(0))
                
                # record loss
                pi_losses.update(l_pi.item(), seq2nums.size(0))
                v_losses.update(l_v.item(), seq2nums.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, sequence):
        """
        sequence: peptide string
        """
        # timing
        start = time.time()
        # preparing input
        seq2num = torch.tensor(Embedding_CLS([sequence]))
        
        if args.cuda:
            seq2num = seq2num.cuda()
        
        self.nnet.eval()
        with torch.no_grad():
            v = self.nnet.ProbLayer(seq2num)
            pi = self.nnet.PIlayer(seq2num)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


rlist = list('ACDEFGHIKLMNPQRSTVWY')
       
def Embedding_CLS(seqlist): # to be modified
  rdict = {}
  for i in range(len(rlist)):
    rdict[rlist[i]] = i
  data = np.array([[20] + [rdict[r] for r in s] for s in seqlist])
  return data