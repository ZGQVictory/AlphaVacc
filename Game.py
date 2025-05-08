class Game():
    """
    This class specifies the base Game class on the basis of the original AlphaGo Zero Framework.
    But here we write some additional functions and make some adjustments about the comments.
    
    Here for the neoantigen framework, we only have one player. 


    In the MCTS, the functions we need are below.
    The function we have changed:
    
    stringRepresentation, getGameEnded, getValidMoves, getActionSize, ActionNum, getNextState
    
    The function we need to change:
    
    getCanonicalForm
    
    In the Coach, the additional functions we need are below.
    The function we need to change:
    
    getSymmetries (perhaps useless)
    
    
    
    """
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
            
            0~180?
            
            here, the number of the mutation residues in this step
        """
        pass

    def getNextState(self, board, action, player = 1):
        """
        Input:
            board: current board
            player: current player (1 or -1) here only 1
            action: action taken by current player 
            Now we just act for one time.

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
            
            # nextPlayer should be eliminated for neoantigen framework
        """
        pass

    def getValidMoves(self, board, history_board, T, step, db, scorefunc, player=1):
        """
        Input:
            board: current board
            player: current player
            history_board: all the history boards (including board)
            T: simulated annealing temperature
            step: episode step
            db: need for the PNADORA
            scorefunc: our loss function for the accepted peptides
        Notice: board == history_board[-1]
        
        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
                        
            # the length should be 180?
            
            Invalid: the state sequence remains unchanged or changes to a previously generated sequence
            Moreover, if there is some additional restrains here, it should be added in the argument
        """
        pass

    def getGameEnded(self, board, step, player=1):
        """
        Input:
            board: current board
            it: iteration number in one episode
            player: current player (1 or -1), this is for the alphago zero two-player game
            For us, we only have one player, so we can only set the player to be 1 by default

        Returns:
            r: 0 if game has not ended. 1 if the game won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

    def getCanonicalForm(self, board, player=1):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
                       
        Notice, this symmetries are written here, trying to give the AI more available information to enlarge the database. 
        Intrinsically, when do the symmetries, all data still the same, but for our neoantigen framework, it seems useless.
        
        WE DON'T USE THIS FUNCTION!
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of list format to a string format of one peptide sequence.
                         Required by MCTS for hashing.
        """
        pass
    
################ WE ADDED ######################################
    def ActionNum(self, step):
        """
        Input:
            step: current optimization step number
        
        Returns:
            mutation_num: the number of the mutation residues in this step. (Action number in this step)
        """