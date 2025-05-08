import numpy as np
import os
# import mhcflurry
import warnings
warnings.filterwarnings('ignore')

class Board():
    
    def __init__(self, n, residueNum, IEDBtargets = None, args=None):
        self.n = n
        self.res = residueNum
        self.IEDBtargets = IEDBtargets
        self.args = args
        
        self.dictionary = dict(zip(np.linspace(0,19,20,dtype = int), 
                            ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']))
        
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.res
        
        if self._is_empty():
            peptide = self._generate_peptide(self.n)
            self._peptide_to_board(peptide)
        self.pieces = np.array(self.pieces)
        
    def __getitem__(self, index):
        return self.pieces[index]
    
    def to_string(self, board = None):
        peptide_sequence = []
        
        if board is not None:
            board = board.tolist()
            for row in board:
                position = row.index(1) if 1 in row else None
                if position is not None:
                    peptide_sequence.append(self.dictionary[position])
                else:
                    peptide_sequence.append("-")
        else:
            self.pieces = self.pieces.tolist()
            for row in self.pieces:
                position = row.index(1) if 1 in row else None
                if position is not None:
                    peptide_sequence.append(self.dictionary[position])
                else:
                    peptide_sequence.append("-")
                    
        return ''.join(peptide_sequence)
    
    def get_legal_moves(self, history, scorefunc, T, refscore, player=1):
        # Returns all the legal moves
        # Set for one mutation
        moves = []
        valid_peptides = set()
        
        current_board = self.pieces.tolist()

        for i in range(self.n):
            k = current_board[i].index(1) if 1 in current_board[i] else None
            if k == None:
                raise ValueError("Current board invalid!")
            for j in range(self.res):
                if j == k:
                    continue
                # Create a copy of the current board to mutate
                mutated_board = [row[:] for row in current_board]
                # Set the kth element in the position to 0
                mutated_board[i][k] = 0
                # Set the jth element to 1
                mutated_board[i][j] = 1

                if self.to_string(np.array(mutated_board)) not in history:
                    moves.append([i, j])
                    
                    # Collect this peptide
                    if self.args.refpeptidestring:
                        bboard = Board(self.n)
                        bboard.pieces = np.copy(mutated_board)
                        s = bboard.to_string()
                        valid_peptides.add(s)
        
        if self.args.refpeptidestring:
            s = list(valid_peptides)
            self._peptide_candidates_collect(s, self.args.refpeptidestring, scorefunc, T, refscore)
                     
        return list(moves)

    def has_iedb_peptide(self, mhcflurry = False, score_barrier = 0.9):
        
        pep = self.to_string()
        if mhcflurry:
            if pep in self.IEDBtargets or self.mhcflurry_test(pep) > score_barrier:
                return True
            else:
                return False
        else:
            if pep in self.IEDBtargets:
                return True
            else:
                return False
        
    def mhcflurry_test(self, pep, allele = "A*02:01"):
        predictor = mhcflurry.Class1PresentationPredictor.load()
        if type(pep) == list:
            results = predictor.predict(pep, [allele])
        else:
            results = predictor.predict([pep], [allele])
        
        return results.presentation_score[0] # return the presentation score
    
    def execute_move(self, move, player = 1):
        x = move[0]
        y = move[1]
        self.pieces = self.pieces.tolist()
        k = self[x].index(1) if 1 in self[x] else None
        assert k != y
        
        self[x][k] = 0
        self[x][y] = 1
        self.pieces = np.copy(self.pieces)
        
    def _is_empty(self):
        for row in self.pieces:
            if any(row):  # Check if any element in the row is non-zero
                return False
        return True

    def _generate_peptide(self, length):
        if self.args.startpeptide:
            assert length == len(self.args.startpeptide)
            return self.args.startpeptide
        else:
            return ''.join(np.random.choice(list(self.dictionary.values()), length))
        
    def _peptide_to_board(self, peptide):
        # Initialize the board with 0
        self.pieces = [[0]*self.res for _ in range(self.n)]
    
        for i in range(self.n):
            try:
                position = list(self.dictionary.keys())[list(self.dictionary.values()).index(peptide[i])]
                self.pieces[i][position] = 1
            except ValueError:
                # Amino acid not found in the dictionary
                raise ValueError(f"Amino acid {peptide[i]} not found in the dictionary.")
        return self.pieces
        
    def _peptide_candidates_collect(self, peptidestring, refpeptidestring, scorefunc, T, refscore):
        # Using metropolis criterion to collect
        # peptidestring might be a list of peptidestrings
        
        if peptidestring == refpeptidestring:
            return 2
        
        pepscoredic = scorefunc.function(peptidestring)  # dictionary of peptidestrings and scores
        
        
        
        if len(pepscoredic) > 1:
            for i, peps in enumerate(pepscoredic.keys()):
                assert len(refpeptidestring) == 1 # only have one refpeptidestring
                
                deltascore = pepscoredic[peps] - refscore
                
                if 1 < np.exp (-deltascore / T):
                    self._write_peptide_to_file(peps, self.args.acceptedpeptide_dir, self.args.acceptedpeptide_file)
                    return 1
                else:
                    if np.random.uniform(0,1) < np.exp (-deltascore / T):
                        self._write_peptide_to_file(peps, self.args.acceptedpeptide_dir, self.args.acceptedpeptide_file)
                        return 1
                    else:
                        return 0
        
        else:
            assert len(refpeptidestring) == 1 # only have one refpeptidestring
            
            deltascore = list(pepscoredic.values())[0] - refscore
        
            if 1 < np.exp (-deltascore / T):
                self._write_peptide_to_file(peptidestring=list(pepscoredic.keys())[0], 
                                                file_dir=self.args.acceptedpeptide_dir, file_str=self.args.acceptedpeptide_file)
                return 1
            else:
                if np.random.uniform(0,1) < np.exp (-deltascore / T):
                    self._write_peptide_to_file(peptidestring=list(pepscoredic.keys())[0], 
                                                file_dir=self.args.acceptedpeptide_dir, file_str=self.args.acceptedpeptide_file)
                    return 1
                else:
                    return 0
            
    def _write_peptide_to_file(self, peptidestring, file_dir, file_str):
        filename = os.path.join(file_dir, file_str)
        mode = 'a' if os.path.exists(filename) else 'w'
        with open(filename, mode) as f:
            f.write(peptidestring + '\n')