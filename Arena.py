import logging
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, args, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.args = args

    def playGame(self, player, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """

        board = self.game.getInitBoard()
        history = []
        it = 0
        while self.game.getGameEnded(board, step = it) == 0:
            history.append(self.game.stringRepresentation(board))
            it += 1
            T = self.args.T_init * (np.exp(np.log(0.5) / self.args.half_life) ** it)
            if verbose:
                assert self.display
                print("") # Need to be updated
                self.display(board)

            action = player(board, history)
            valids = self.game.getValidMoves(board, history, T, it)
            
            if valids[action] == 0:
                # usually, the count list elements within the MCTS_mut.py are all zero.
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0

                
            board = self.game.getNextState(board, action)
        
        if verbose:
            assert self.display
            print("Game over: ") # Need to be updated
            self.display(board)
        
        return it, self.game.getGameEnded(board, step = it)


    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):

            it1, gameResult1 = self.playGame(player = self.player1, verbose=verbose)
            it2, gameResult2 = self.playGame(player = self.player2, verbose=verbose)
            
            if gameResult1 == gameResult2:
                if gameResult1 == 1: # both win, compare iteration numbers
                    if it1 < it2:
                        oneWon += 1
                    elif it1 > it2:
                        twoWon += 1
                    else:
                        draws += 1
                else:
                    draws += 1
            elif gameResult1 > gameResult2:
                oneWon += 1
            else:
                twoWon += 1
            

        return oneWon, twoWon, draws