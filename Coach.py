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

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self, playtoend=False):
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
            
            if playtoend:
                r = self.game.getGameEnded(canonicalBoard, episodeStep, nnet = False, player=self.curPlayer)
            else:
                r = self.game.getGameEnded(canonicalBoard, episodeStep, self.nnet, self.curPlayer)
            
            if r != 0:
                return [(x[0], x[2], r) for x in trainExamples]
            
            temp = int(episodeStep < self.args.tempThreshold)
            history.append(self.game.stringRepresentation(canonicalBoard))
            
            if playtoend:
                if episodeStep == 1:
                    for ff in os.listdir():
                        # Check if the file starts with "Manualrecord"
                        if ff.startswith("Manualrecord"):
                            break
                    assert ff.startswith("Manualrecord")
                    
                    ff = open(ff, 'a')
                    print("NOW MOVES TO PLAYTOEND", file = ff)
                    ff.close()
                
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

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration for self play
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue) # restrain the maximum length 

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):# self play for self.args.numEps episodes
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    if _ < (1 - self.args.playtoend) * self.args.numEps:
                        iterationTrainExamples += self.executeEpisode()
                    else:
                        iterationTrainExamples += self.executeEpisode(playtoend=True)

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)
            
            # ready for the training of the neural network
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            log.info('The Length of trainExamples is %s', len(trainExamples))
            # 每一轮iteration结束，训练一下神经网络。然后对训练前后的神经网络，各自利用MCTS，在Arena中相互博弈
            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            
            for ff in os.listdir():
                # Check if the file starts with "Manualrecord"
                if ff.startswith("Manualrecord"):
                    break
            assert ff.startswith("Manualrecord")
                    
            ff = open(ff, 'a')
            print("NOW MOVES TO Arena PROCESS", file = ff)
            ff.close()
            
            arena = Arena(lambda x, y: np.argmax(pmcts.getActionProb(x, y, temp=0, arena=True)),
                          lambda x, y: np.argmax(nmcts.getActionProb(x, y, temp=0, arena=True)), self.game, self.args)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                
                for ff in os.listdir():
                    # Check if the file starts with "Manualrecord"
                    if ff.startswith("Manualrecord"):
                        break
                assert ff.startswith("Manualrecord")
                    
                ff = open(ff, 'a')
                print("NOW We GOT A NEW MODEL", file = ff)
                ff.close()
                
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True