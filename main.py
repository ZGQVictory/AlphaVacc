import logging
import os 
import coloredlogs

from Coach import Coach
from peptideMutGame import peptideMutGame as Game
from peptideMutLogic import Board
from NNetwrapper import NNetWrapper as nn
from utils import *



log = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

args = dotdict({
    'pep_length': 9,
    'res_type':20,
    
    'numIters': 100,             # We have already done 36 rounds
    'numEps': 200,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 0.3,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp/','checkpoint_11.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    
    
    'MaxIterinONEepisode': 100,   # zgq added
    'playtoend':0.1, # set the threshold for self-play to the end  
    'mutation_rate':'3-1',
    'startpeptide': None,
    'refpeptidestring': None,
    'acceptedpeptide_dir':'../data/',
    'acceptedpeptide_file':'peptide_candidates.txt',
    'half_life':500,
    'T_init':20,
    'refscore': None,
    'IEDBdir': './data/IEDB',
    'IEDBtargetdatabase':'IEDB-target-9res.txt',
})

# args 在最初始要再额外计算更改args.refscore

# refscoredic = scorefunc.function(refpeptidestring) # dictionary of refpeptidestring and score
# refscore = list(refscoredic.values())[0]
# Scorefunc().db = "......."

def main():
    log.info('Loading %s...', Game.__name__)
    create_and_update_start_record_files()
    create_and_update_chess_manual_files()
    
    # Generate the initial IEDB target
    IEDBtargets = read_peptides(args.IEDBdir, args.IEDBtargetdatabase)
    
    g = Game(args, IEDBtargets)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

def read_peptides(dir, filename):
    filename = os.path.join(dir, filename)
    sequences = []
    with open(filename, 'r') as file:
        # Skip the first line
        next(file)
        for line in file:
            # Extract the sequence, assuming it's the only element in each line
            sequence = line.strip()
            if len(sequence) == 9:  # Ensuring the sequence length is 9
                sequences.append(sequence)
    return sequences


import datetime

def create_and_update_start_record_files():
    # Get the current date in 'YYYYMMDDHHMM' format (Year-Month-Day-Hour-Minute)
    current_date = datetime.datetime.now().strftime('%Y%m%d%H%M') 

    # Construct the new file name
    new_file_name = f"Startrecord-{current_date}.txt"

    # Get a list of all files in the current directory
    for file in os.listdir():
        # Check if the file starts with "Startrecord"
        if file.startswith("Startrecord") and file != new_file_name:
            # Rename the file by prefixing it with "OLD-"
            os.rename(file, f"OLD-{file}")

    # Check if the new file exists
    if not os.path.exists(new_file_name):
        # Create the file if it doesn't exist
        with open(new_file_name, 'w') as file:
            file.write(f"Start record for {current_date}\n")
    else:
        print(f"The file '{new_file_name}' already exists.")

def create_and_update_chess_manual_files():
    # Get the current date in 'YYYYMMDDHHMM' format (Year-Month-Day-Hour-Minute)
    current_date = datetime.datetime.now().strftime('%Y%m%d%H%M') 

    # Construct the new file name
    new_file_name = f"Manualrecord-{current_date}.txt"

    # Get a list of all files in the current directory
    for file in os.listdir():
        # Check if the file starts with "Manualrecord"
        if file.startswith("Manualrecord") and file != new_file_name:
            # Rename the file by prefixing it with "OLD-"
            os.rename(file, f"OLD-{file}")

    # Check if the new file exists
    if not os.path.exists(new_file_name):
        # Create the file if it doesn't exist
        with open(new_file_name, 'w') as file:
            file.write(f"Manual record for {current_date}\n")
    else:
        print(f"The file '{new_file_name}' already exists.")

if __name__ == "__main__":
    main()

