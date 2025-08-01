import logging
import math

import numpy as np
import os
EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        

    def getActionProb(self, canonicalBoard, history=[], temp=1, arena = False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        arena means that the game ended is not determined by the predicted value 
        of the neural network.
        
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        mcts_history = history.copy()
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard, mcts_history, step = 0, arena = arena)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        
        if temp == 0:
            # bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            # bestA = np.random.choice(bestAs)
            # In order to avoid select the wrong moves
            valids = self.game.getValidMoves(canonicalBoard, history)
            
            # while valids[bestA] == 0:
            #     if any(_ > 0 for _ in  counts):
            #         raise ValueError(f"Here the counts is {counts}, Not all the elements are all zero. Check the details.")
            #     else:
            #         print(f"All the elements in counts are zero. Now, the peptide is {s}. Please check if it is in the database.")
            #         print("Select best action for another round.")
            #         bestA = np.random.choice(bestAs)
            # We found that the bestA miht also be the invalid option       
            
            while True:
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)
                if valids[bestA] != 0:
                    break
                counts[bestA] = 0
                    
            probs = [0] * len(counts)
            probs[bestA] = 1
            
            if arena: # Record the chess manual
                for ff in os.listdir():
                    # Check if the file starts with "Manualrecord"
                    if ff.startswith("Manualrecord"):
                        break
                assert ff.startswith("Manualrecord")
                
                ff = open(ff, 'a')
                print("When meeting {}".format(s), file = ff)
                print("The counts are {}".format(counts), file = ff)
                print("The bestAs are {}".format(bestAs), file = ff)
                print("The probability matrix searched by MCTS is:", file = ff)
                print(probs,"\n", file=ff)
                ff.close()
            
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]

        if arena: # Record the chess manual
            for ff in os.listdir():
                # Check if the file starts with "Manualrecord"
                if ff.startswith("Manualrecord"):
                    break
            assert ff.startswith("Manualrecord")
            
            ff = open(ff, 'a')
            print("When meeting {}".format(s), file = ff)
            print("The probability matrix searched by MCTS is:", file = ff)
            print(probs,"\n", file=ff)
            ff.close()
            
        return probs

    def search(self, canonicalBoard, history, step, arena = False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
            
        For the alphago zero, the value of the current canonicalBoard considered is the state that the other player just done, 
        so we need the negative.
        But here, we only have one player, so the value keeps the same 
        """
        step += 1
        
        T = self.args.T_init * (np.exp(np.log(0.5) / self.args.half_life) ** step)
        s = self.game.stringRepresentation(canonicalBoard)
        
        history = history.copy()
        history.append(s)
        # print("MCTS Step", step)
        # MCTS part
        if s not in self.Es:
            if arena:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, step)
            else:
                self.Es[s] = self.game.getGameEnded(canonicalBoard, step, self.nnet) # canonicalBoard should consist of two things: the state and the step number.

        if self.Es[s] != 0:
            # terminal node
            return self.Es[s] # remove the minus sign, used to be -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(s)
            return v # remove the minus sign, used to be -v

        valids = self.game.getValidMoves(canonicalBoard, history, T, step) 
        self.Ps[s] = self.Ps[s] * valids  # masking invalid moves

        sum_Ps_s = np.sum(self.Ps[s])
        
        if sum_Ps_s > 0:
            self.Ps[s] /= sum_Ps_s  # renormalize
        else:
            # if all valid mutations were masked make all valid mutations equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
            log.error("All valid mutations were masked, doing a workaround.")
            self.Ps[s] = self.Ps[s] + valids
            self.Ps[s] /= np.sum(self.Ps[s])

        self.Vs[s] = valids
        self.Ns[s] = 0
        
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        
        # Here we should mutate several residues of a peptide in one step, so there should also be another function that calculates the number of the mutations
        # for _ in range(self.game.ActionNum):
        #     best_act_list = []
            # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()): # getActionSize should be restrained by the valids?
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a
            # best_act_list.append(best_act)

        # a = best_act_list # here a should be a list representing the mutation residues
        a = best_act
        next_s = self.game.getNextState(canonicalBoard, a)
        next_s = self.game.getCanonicalForm(next_s)
        
        v = self.search(next_s, history, step, arena)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1) # mean
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v # change -v to v
    
