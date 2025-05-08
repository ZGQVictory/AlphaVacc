from pickle import Pickler, Unpickler
import sys

file_name = sys.argv[1]
with open(file_name, 'rb') as f:
    a = Unpickler(f).load()

with open(file_name.split('.')[0] + '_training_sequences_withZ.txt', 'w') as f:
    for i in a:
        for ele in i:
            print(ele[0], end = ' ', file = f)
            print(ele[2], end = '\n', file = f)
