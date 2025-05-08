import logging
import random
import sys
import os
import collections
import numpy as np
import torch
from torch import nn

# from Structure_generation import predict_structure
# from Loss import compute_loss
# from addition_func import *
# from Score_NNet import RNN_model

log = logging.getLogger(__name__)

class Scorefunc():
    
    def __init__(self, step, functype = 0):
        
        self.step = step
        self.functype = functype
        self.db = None
        
        ########### Changing Part ##############################
        dir_tcr = '/public/home/zhangguanqiao/data/pmhc/' ###change 1###
        file_tcr = 'tcr-specific--peptide.txt'
        # IEDB——HLA0201-Database
        dir_iedb = '/public/home/zhangguanqiao/data/pmhc'  ###Change 2###
        file_iedb = 'pHLA-A0201--peptide.txt'
        # PANDORA Init
        ###Change 3 in the main function###
        ########################################################
        
        self.iedb_freqmat, num_pep = get_freq(dir_iedb, file_iedb, 9)
        self.prob_ref_freqmat, num_pep = get_freq(dir_tcr, file_tcr, 9)
        
        if self.functype == 1:
            self.function =  self.knowledge_based_function
        elif self.functype == 2:
            self.function = self.nnet_based_function
        else:
            self.function = self.com_function
        
    def knowledge_based_function(self, peptidestring):
        # peptidestring may be a list of strings or one single string.
        # should return a dictionary, key is the sequences, value is the scores
        
        db = self.db
        if type(peptidestring) == str:
            numpeptide = 1
        else:
            numpeptide = len(peptidestring)
            
        if numpeptide == 1:
            out_filename = 'TestCase_Step' + str(self.step) + peptidestring
            
        else:
            out_filename = []
            for nmt in range(numpeptide):
                out_filename.append('TestCase_Step' + str(self.step) + '_NUM' + str(nmt+1))
                
        # construct the structures        
        try:
            predict_structure(dict(zip(np.linspace(0,8,9,dtype = int), peptidestring)), 
                              out_filename, db=db, mutant_times = len(peptidestring))
        except:
            sys.exit('Step {} collapsed!!!'.format(self.step))
        
        # calculate the loss
        if numpeptide == 1:
            loss = compute_loss(self.iedb_freqmat, self.prob_ref_freqmat, out_filename, list(peptidestring))
            return {peptidestring: loss}
        
        else:
            loss_target = []
            for r in range(numpeptide):
                loss = compute_loss(self.iedb_freqmat, self.prob_ref_freqmat, out_filename[r], list(peptidestring[numpeptide]))
                loss_target.append(loss)
            return dict(zip(peptidestring, loss_target))
    
    def nnet_based_function(self, peptidestring):
        # peptidestring may be a list of strings or one single string.
        # should return a dictionary
        
        if type(peptidestring) == str:
            numpeptide = 1
        else:
            numpeptide = len(peptidestring)
        
        net = RNN_model()
        net.load_state_dict(torch.load('../data/RNN.params'))
        net.eval()
        
        directory = '../data/'
        file = 'databank_test.log' #有点不严谨
        lines = self._read_sequence(directory, file)
        tokens = self._tokenize(lines)
        tokens = [list(_) for _ in np.array(tokens)[:, 1]]
        vocab = Vocab(tokens)
        
        if numpeptide == 1:
            return dict(zip(peptidestring, torch.exp(self._nnet_predict_single(pep, net, vocab, 'cuda:0'))))
        else:
            loss_target = []
            for pep in peptidestring:
                loss_target.append(torch.exp(self._nnet_predict_single(pep, net, vocab, 'cuda:0')))
            return dict(zip(peptidestring, loss_target))
        
    
    def com_function(self, peptidestring):
        # peptidestring may be a list of strings or one single string.
        step = self.step
        
        if step % 100 == 0:
            if len(peptidestring) < 20:
                return self.knowledge_based_function(peptidestring)
            else:
                # Randomly select 20 peptides
                selected_peptides = random.sample(peptidestring, 20)
                log.info("Selected Peptides:")
                log.info(selected_peptides)
                log.info("for the knowledge_based_function.")
                
                
                # Get peptides that are not in the selected list
                remaining_peptides = [pep for pep in peptidestring if pep not in selected_peptides]
                
                log.info("\nRemaining Peptides:")
                log.info(remaining_peptides)
                log.info("for the nnet_based_function.")
                
                return {**self.knowledge_based_function(selected_peptides), **self.nnet_based_function(remaining_peptides)}

        else:
            return self.nnet_based_function(peptidestring)
            
    def _nnet_predict_single(seq, net, vocab, device):
        net.eval()
        state = net.begin_state(batch_size=1, device = device)
        seq = torch.tensor(vocab[[i for i in seq]], device = device).unsqueeze(0)
        print(seq)
        y, _ = net(seq,state)
        return y

    def _read_sequence(directory, file):
        with open(os.path.join(directory, file), 'r') as f:
            return f.readlines()
        
    def _tokenize(lines):
        alltoken = []
        # 读取这个文件，并将数字和seqeunce内容分开
        for line in lines:
            if list(line)[0] == '#':
                continue
            else:
                alltoken.append(line.strip().split())
        return alltoken



class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        counter = counter_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
        
        

def counter_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    
    return collections.Counter(tokens)