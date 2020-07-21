import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn
import gc
import sklearn
import pickle
from tqdm import tqdm
#from difflib import SequenceMatcher
from multiprocessing import Value, Pool, Lock
from multiprocessing import Manager
import time
import Levenshtein
from itertools import chain

    
def similar(a, b):
    #return SequenceMatcher(None, a, b).ratio()
    return Levenshtein.distance(a, b) / min(len(a), len(b))
    

def thefunction(i):
    with lock:
        print("Working on: " + str(i) + "/" + str(len(seqs)))
        print(" ")
    string1 = seqs[i]
    for j in range(i+1,len(seqs),1):
        string2 = seqs[j]
        if similar(string1, string2) <= 0.25:
            with lock:
                #print('Sequence idx ' + str(i) + ' and sequence idx ' + str(j) + ' are highly similar!')
                temp = ns.sim_array
                temp[i][j] = 1
                ns.sim_array = temp
     
def initializer(*args):
    global lock, ns, seqs
    lock, ns, seqs = args
    
def parse_fasta_append (lines, seqs=[]):
    descs = []
    data = ''
    for line in lines:
        if line.startswith('>'):
            if data:   # have collected a sequence, push to seqs
                seqs.append(data)
                data = ''
            descs.append(line[1:])  # Trim '>' from beginning
        else:
            data += line.rstrip('\r\n')
    # there will be yet one more to push when we run out
    seqs.append(data)
    length = len(descs)
    return descs, seqs, length
    
def extract_pos():

    pos_dirpath = './Pfam_fasta/positive'
    print('Available pos files: ', os.listdir(pos_dirpath))
            
    pos_seqs = []
    pos_labels = []
    for fn in os.listdir(pos_dirpath):
        handle = open(os.path.join(pos_dirpath, fn), 'r')
        descs, pos_seqs, num_seq = parse_fasta_append(handle.read().split('\n'), pos_seqs)
        labs = [fn]*num_seq
        pos_labels.append(labs)
        handle.close()
        
    pos_labels = list(chain.from_iterable(pos_labels))
        
    #need to get rid of alignment since using that would be cheating

    pos = []

    for seq in pos_seqs:
        s = seq.replace("-", "")
        s = s.replace(".", "")
        s = s.upper()
        pos.append(s)
        
    return pos, pos_labels
    
def extract_neg():

    neg_dirpath = './Pfam_fasta/negative'

    neg_seqs = []
    neg_labels = []
    for fn in os.listdir(neg_dirpath):
        handle = open(os.path.join(neg_dirpath, fn), 'r')
        descs, neg_seqs, num_seq = parse_fasta_append(handle.read().split('\n'), neg_seqs)
        labs = [fn]*num_seq
        neg_labels.append(labs)
        handle.close()
        
    neg_labels = list(chain.from_iterable(neg_labels))
    
    neg = []

    for seq in neg_seqs:
        s = seq.replace("-", "")
        s = s.replace(".", "")
        neg.append(s)
    
    return neg, neg_labels
    
def reduce(seqs, labels, prepend):

    manager = Manager()
    ns = manager.Namespace()
    siz = len(seqs)
    ns.sim_array = [[0] * siz] * siz
    lock = Lock()	         
    numcores = 7
    siz = len(seqs)
    pool = Pool(numcores, initializer, (lock, ns, seqs))
    s_parallel = time.time()
    pool.map(thefunction, range(len(seqs)))
    e_parallel = time.time()
    print("Total time to remove similar sequences: " + str(e_parallel - s_parallel))
    
    deleted = []
    contact = np.array(ns.sim_array)

    while np.sum(np.sum(contact, 0)) != 0:
        idx_to_be_deleted = np.argmax(contact)
        contact = np.delete(contact, idx_to_be_deleted, 0)
        contact = np.delete(contact, idx_to_be_deleted, 1)
        deleted.append(seqs[idx_to_be_deleted])
        del seqs[idx_to_be_deleted]
        del labels[idx_to_be_deleted]
        
    print("Number of deleted positive sequences: " + str(len(deleted)))
    print("Number of reduced positive sequences: " + str(len(seqs)))

    path = "./Pfam_fasta/" + prepend + "_seqs_reduced" + ".pickle"
    output = open(path, 'w+b')
    pickle.dump(seqs, output)
    output.close()

    path = "./Pfam_fasta/" + prepend + "_labels_reduced" + ".pickle"
    output = open(path, 'w+b')
    pickle.dump(labels, output)
    output.close()
    
if __name__ == '__main__':
    seqs, labels = extract_pos()
    reduce(seqs, labels, "pos")
    seqs, labels = extract_neg()
    reduce(seqs, labels, "neg")
    

    

    

    
    