import logging
import os 
import coloredlogs

from Coach import Coach
from peptideMutGame import peptideMutGame as Game
from peptideMutLogic import Board
from NNetwrapper import NNetWrapper as nn
from utils import *
from Search import Search

import sys
import datetime

checkpointmodel = sys.argv[1]

log = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

args = dotdict({
    'pep_length': 9,
    'res_type':20,
    
    'numIters': 100,
    'numEps': 200,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 200,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 0.3,

    'checkpoint': './temp',
    'load_model': True,
    'load_folder_file': ('./temp', checkpointmodel),
    'predict_directory':"./Predict_data/<peptide>".format(checkpointmodel),     # Directory where the files are located
    'numItersForTrainExamplesHistory': 20,
    
    
    'MaxIterinONEepisode': 200,   # zgq added
    'playtoend':0.1, # set the threshold for self-play to the end  
    'mutation_rate':'3-1',
    'startpeptide': '<peptide>',
    'refpeptidestring': None,
    'acceptedpeptide_dir':None,
    'acceptedpeptide_file':None,
    'half_life':500,
    'T_init':20,
    'refscore': None,
    'IEDBdir': './data/IEDB',
    'IEDBtargetdatabase':'IEDB-target-9res.txt',

    # predict parameters
    'optimizationSTEP':50,
    'MaxIterinONEepisode': 1000, 
    'Use_MCTS': True
})

# args 在最初始要再额外计算更改args.refscore

# refscoredic = scorefunc.function(refpeptidestring) # dictionary of refpeptidestring and score
# refscore = list(refscoredic.values())[0]
# Scorefunc().db = "......."

def main():
    start_time = datetime.datetime.now()
    create_and_update_predict_start_record_files(args)
    
    if args.Use_MCTS:
        create_and_update_start_record_files()
        create_and_update_chess_manual_files()

    updatingHistory = []
    log.info('Loading %s...', Game.__name__)
    
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
    s = Search(g, nnet, args)

    log.info('Starting the updating process 🎉')
    if args.Use_MCTS:
        # Using MCTS
        updatingHistory += s.executeEpisode()
    else:
        # Using Neural Net directly
        updatingHistory += s.predict_usingNN(optimizationSTEP=args.optimizationSTEP)
    
    for ff in os.listdir(args.predict_directory):
        # Check if the file starts with "update-..."
        if ff.startswith(f"update-{args.load_folder_file[1]}"):
            break
    assert ff.startswith(f"update-{args.load_folder_file[1]}")
    ff = os.path.join(args.predict_directory, ff) 
    ff = open(ff, 'a')
    winorloss = -1
    for i in updatingHistory:
        print(i[0], file = ff)
        # if i[2] == 1:
        #     winorloss = 1
    # if not args.Use_MCTS:
    #     if winorloss == 1:
    #         print("# Finally, we won", file=ff)
    #     else:
    #         print("# Finally, we lost", file=ff)
        
    ff.close()
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)

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
    current_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S') 

    # Construct the new file name
    new_file_name = f"Startrecord-{current_date}.txt"

#    # Get a list of all files in the current directory
#    for file in os.listdir():
#        # Check if the file starts with "Startrecord"
#        if file.startswith("Startrecord") and file != new_file_name:
#            # Rename the file by prefixing it with "OLD-"
#            os.rename(file, f"OLD-{file}")

    # Check if the new file exists
    if not os.path.exists(new_file_name):
        # Create the file if it doesn't exist
        with open(new_file_name, 'w') as file:
            file.write(f"Start record for {current_date}\n")
    else:
        print(f"The file '{new_file_name}' already exists.")

def create_and_update_chess_manual_files():
    # Get the current date in 'YYYYMMDDHHMM' format (Year-Month-Day-Hour-Minute)
    current_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S') 

    # Construct the new file name
    new_file_name = f"Manualrecord-{current_date}.txt"

#    # Get a list of all files in the current directory
#    for file in os.listdir():
#        # Check if the file starts with "Manualrecord"
#        if file.startswith("Manualrecord") and file != new_file_name:
#            # Rename the file by prefixing it with "OLD-"
#            os.rename(file, f"OLD-{file}")

    # Check if the new file exists
    if not os.path.exists(new_file_name):
        # Create the file if it doesn't exist
        with open(new_file_name, 'w') as file:
            file.write(f"Manual record for {current_date}\n")
    else:
        print(f"The file '{new_file_name}' already exists.")

def create_and_update_predict_start_record_files(args):
    # Ensure the directory exists
    os.makedirs(args.predict_directory, exist_ok=True)

    # Get the current date in 'YYYYMMDDHHMM' format (Year-Month-Day-Hour-Minute)
    current_date = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Construct the new file name
    if args.startpeptide:
        new_file_name = f"update-{args.load_folder_file[1]}-{args.startpeptide}-{current_date}.txt"
    else:
        new_file_name = f"update-{args.load_folder_file[1]}-{current_date}.txt"
    
    # Full path of the new file
    new_file_path = os.path.join(args.predict_directory, new_file_name)

#    # Get a list of all files in the specified directory
#    for file in os.listdir(args.predict_directory):
#        # Check if the file starts with "update"
#        if file.startswith(f"update-{args.load_folder_file[1]}") and file != new_file_name:
#            # Rename the file by prefixing it with "OLD-"
#            os.rename(os.path.join(args.predict_directory, file), os.path.join(args.predict_directory, f"OLD-{file}"))

    # Check if the new file exists
    if not os.path.exists(new_file_path):
        # Create the file if it doesn't exist
        with open(new_file_path, 'w') as file:
            file.write(f"# Start updating process for {current_date}\n")
    else:
        print(f"The file '{new_file_name}' already exists.")
        with open(new_file_path, 'a') as file:
            file.write(f"# Start updating process for {current_date}\n")



if __name__ == "__main__":
    main()

