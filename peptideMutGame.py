from Game import Game
from peptideMutLogic import Board
from NNetwrapper import NNetWrapper as nn

from Score import Scorefunc
import numpy as np
import os

class peptideMutGame(Game):
    def __init__(self, args, IEDBtargets):
        self.args = args
        self.IEDBtargets = IEDBtargets
        
        self.n = self.args.pep_length # n is the length of the peptide
        self.res = self.args.res_type
        
    def getInitBoard(self, scorefunc=Scorefunc):
        b = Board(self.n, self.res, self.IEDBtargets, self.args)
        return b.pieces
        
    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.res) # 20 amino acids
    
    def getActionSize(self):
        # return the number of all the amino acids in each position
        return self.n * self.res
    
    def getNextState(self, board, action, player=1):
        # we only have one player, thus set player = 1
        b = Board(self.n, self.res, self.IEDBtargets, self.args)
        b.pieces = np.copy(board)
        
        # move = [(int(act/self.res), act%self.res) for act in action] # move --> [(pos1, res1), (pos2, res2), (pos3, res3)...]
        # # Caution! each element in the move should be different
        # for m in move:
        #     b.execute_move(move, player)
        
        move = (int(action/self.res), action%self.res)
        b.execute_move(move, player)
            
        return b.pieces # player don't change
    
    def getValidMoves(self, board, history, T = None, step = None, db = None, scorefunc=Scorefunc, player=1):
        # return a fixed size binary vector
        
        # scorefunc = scorefunc(step)
        # scorefunc.db = db
        valids = [0] * self.getActionSize()
        b = Board(self.n, self.res, self.IEDBtargets, self.args)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(history, scorefunc, T, self.args.refscore)
        if len(legalMoves) == 0:
            return np.array(valids) # all 0.
        for x, y in legalMoves:
            valids[self.res*x + y] = 1
        return np.array(valids)
    
    def getGameEnded(self, board, step, nnet = False, player=1, score = 0.95):
        # 1 if player won, 0 if player lost
        for ff in os.listdir():
            # Check if the file starts with "Startrecord"
            if ff.startswith("Startrecord"):
                break
        ff = open(ff, 'a')
        
        b = Board(self.n, self.res, self.IEDBtargets, self.args)
        b.pieces = np.copy(board)
        s = self.stringRepresentation(board)
        if nnet:
            _, v = nnet.predict(s)
            if b.has_iedb_peptide() and step <= self.args.MaxIterinONEepisode:
                print("!!Find the real peptide in the IEDBtargets!! {}".format(s), file = ff)
                ff.close()
                return 1
            elif step > self.args.MaxIterinONEepisode:
                return -1
            
            elif v > score and step <= self.args.MaxIterinONEepisode:
                print("Seems Won in Neural Net! {}".format(s), file = ff)
                ff.close()
                return 1
            elif v < - score and step <= self.args.MaxIterinONEepisode:
                return -1
            
            else:
                return 0
        else:
            
            if b.has_iedb_peptide() and step <= self.args.MaxIterinONEepisode:
                print("!!Find the real peptide in the IEDBtargets:\n{}".format(s), file = ff)
                ff.close()
                return 1
            elif step > self.args.MaxIterinONEepisode:
                return -1
            else:
                return 0
                    
    def getCanonicalForm(self, board, player=1):
        # return state if player==1, else return -state if player==-1
        return player*board
    
    def getSymmetries(self, board, pi):
        # we don't need this function
        pass
    
    def stringRepresentation(self, board):
        b = Board(self.n, self.res, self.IEDBtargets, self.args)
        
        b.pieces = np.copy(board)
        
        return b.to_string()
    
    def ActionNum(self, step):
        # return action number (mutated residues for one times)
        # Now useless
        Mi, Mf = self.args.mutation_rate.split('-')
        M = np.linspace(int(Mi), int(Mf), self.args.MaxIterinONEepisode) 
        
        return round(M[step]) 