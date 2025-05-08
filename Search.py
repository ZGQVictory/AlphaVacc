import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS_mut import MCTS

class Search():
    # This class mainly for the searching tasks using the mcts model that already trained
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
    
    def executeEpisode(self, playtoend=True):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        history = []
        
        while True: # 下一局棋
            
            episodeStep += 1
            # print("Episode Step", episodeStep)
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            
            if playtoend: # When to stop 
                r = self.game.getGameEnded(canonicalBoard, episodeStep, nnet = False, player=self.curPlayer)
            else:
                r = self.game.getGameEnded(canonicalBoard, episodeStep, self.nnet, self.curPlayer)
            
            if r != 0:
                return [(x[0], x[2], r) for x in trainExamples]
            
            temp = int(episodeStep < self.args.tempThreshold)
            history.append(self.game.stringRepresentation(canonicalBoard))
            
            if playtoend:                
                pi = self.mcts.getActionProb(canonicalBoard, history, temp=temp, arena = playtoend)
            else:
                pi = self.mcts.getActionProb(canonicalBoard, history, temp=temp)
            # ##### the symmetry here seems useless for neoantigen framework
            # sym = self.game.getSymmetries(canonicalBoard, pi)
            # for b, p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])
            # ################
            trainExamples.append([self.game.stringRepresentation(canonicalBoard), self.curPlayer, pi, None]) # here there is no "pass" in the policy, which means pi --dimensions--> self.n ** 2

            action = np.random.choice(len(pi), p=pi)
            board = self.game.getNextState(board, action, self.curPlayer)
        
    def predict_usingNN(self, optimizationSTEP):
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        history = []
        canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
        
        while episodeStep < optimizationSTEP:
            episodeStep += 1
            T = self.args.T_init * (np.exp(np.log(0.5) / self.args.half_life) ** episodeStep)
            history.append(self.game.stringRepresentation(canonicalBoard))
            s = self.game.stringRepresentation(canonicalBoard)
            
            r = self.game.getGameEnded(canonicalBoard, episodeStep, nnet = False, player=self.curPlayer)
            
            # predict
            prob, v = self.nnet.predict(s)
            valids = self.game.getValidMoves(canonicalBoard, history, T, episodeStep)
            prob = prob * valids  # masking invalid moves

            sum_prob = np.sum(prob)
            
            if sum_prob > 0:
                prob /= sum_prob  # renormalize
            else:
                # if all valid mutations were masked make all valid mutations equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("Someting wrong in Neural Net, the Neural Net can't predict the valid points.")
                prob = prob + valids
                prob /= np.sum(prob)
            
            trainExamples.append([self.game.stringRepresentation(canonicalBoard), self.curPlayer, prob, r, v]) 
            
            a = np.random.choice(range(self.game.getActionSize()), p=prob)  # get the action through prob
            next_s = self.game.getNextState(canonicalBoard, a)
            canonicalBoard = self.game.getCanonicalForm(next_s)
            
        return [(x[0], x[2], x[3], x[4]) for x in trainExamples]