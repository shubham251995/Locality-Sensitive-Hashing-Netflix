#!/usr/bin/env python
# coding: utf-8


# Import packages
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import *
from random import randint
from itertools import *
from collections import *
import sys
import csv
import math
import time
import random
from argparse import *


parser = ArgumentParser()
parser.add_argument('-d', dest = 'path', help = 'File path', type = str)
parser.add_argument('-s', dest = 'randseed', help = 'Random Seed', default = 2021, type = int)
parser.add_argument('-m', dest = 'method', help = 'Method: js,cs or dcs', default = 'js', type = str)
args = parser.parse_args()

#path = args.path
#randseed = args.randseed
#method = args.method




# Import data
data = np.load(args.path)




def make_csc(data, method=None):
    if method == 'js':
        sparse = csc_matrix((data[:,2], (data[:,0], data[:,1])), shape = (103704,17771))
        # Change all the values to 1's and 0's
        sparse[sparse > 0] = 1
        return sparse
    if method == 'cs':
        sparse = csc_matrix((data[:,2], (data[:,0], data[:,1])), shape = (103704,17771))
        return sparse
    if method == 'dcs':
        sparse = csc_matrix((data[:,2], (data[:,0], data[:,1])), shape = (103704,17771))
        return sparse
    else:
        return print("Pick cosine or jaccard as method")





def make_signature(sparse_data, seed, n_hash, method=None):
    np.random.seed(seed)
    if method == None:
        return print("Please choose a method to create the signature matrix. Choose either js, cs, or dcs")
    
    if method == 'cs' or method == 'dcs':
        ## Add code for permutation ##
        # Making 100 different orders of movies
        permutations = np.array([np.random.permutation(17770) for i in range(n_hash)])
        
        #Empty signature matrix
        signature = np.zeros(shape=(n_hash, 103703), dtype=int)
        
        ##### Fill the signature matrix with random permutations #####
        # 1. find the positions where it's non-zero in the sparse matrix and select columns in permutations using these positions
        # 2. Take the minumum for each selected column in permutations
        # 3. Fill these positions in the signature matrix using the minimums for each selected column n_selected=100

        for index, u in enumerate(sparse_data):
            try:
                signature[:,index] = permutations[:,u.nonzero()[1]].min(axis=1)
            except:
                pass
            
        return signature
        
    if method == 'js':
        
        ### Add code for minhash ###
        
        #Make lists with random vals a and b
        a = []
        b = []
        for i in range(n_hash):
            a_rand = random.randint(0,103704)
            b_rand = random.randint(0,103704)
            if a_rand not in a:
                a.append(random.randint(0,103704))
            if b_rand not in b:
                b.append(random.randint(0,103704))
                
        # Use these random lists to create a list of hashed values in h
        h = []
        nonzero_set = set(sparse.nonzero()[0])
        for row in range(1,103704):
            if row in nonzero_set:
                for col in range(n_hash):
                    h.append((a[col]*row + b[col]) % 103723)
            else:
                pass
        #A matrix of hashed values     
        hashed = np.reshape(h, (n_hash, 103703))
        
        # Empty Signature matrix
        signature = np.zeros(shape=(n_hash,103703), dtype=int)
        
        for index, u in enumerate(sparse_data):
            try:
                signature[:,index] = hashed[:,u.nonzero()[1]].min(axis=1)
            except:
                pass
            
        return signature      
        
    else:
        return print("Please choose a method to create the signature matrix. Choose either js, cs, or dcs")





def LSH(signature, b):

    # Apply LSH
    # input: Signature matrix
    # output: List of potential pairs of users
    
    # Note: I only need to know b since we assume all b*r = n
    # If b*r != n exactly i.e. n=100 and b=3 then np.array_split will make 3 bands of row lengths 34, 33, 33. 


    # Using defaultdict because it is a pretty efficient way to add things to a dict
    # Using set because there are no duplicates in a set and more efficient
    buckets = defaultdict(set)

    # make bands by splitting the signiture matrix into b bands each 
    bands = np.array_split(signature, b)
    for index, band in enumerate(bands):
        for i in range(103703):
            band_num = tuple(list(band[:,i])+[str(index)])
            buckets[band_num].add(i)


        #Now let's make distinct pairs using these buckets
        pairs = set()

        #iterate over the values in buckets to create pairs
        for x in buckets.values():
            if len(x) > 1: 
                for pair in combinations(x, 2):
                    pairs.add(pair)

    return pairs





def jaccard(sparse, signature, candidates):
    pairs = []
    
    for i in candidates:
        # Jaccard is taking way too long to execute. Most of the candidates are trash.
        # Let's make the number of candidates lower based on the signature matrix 
        # Check where the signature of u1 equals u2, then count how many times they are the same over the total number
        # At least 50 percent of the signatures should be the same
        if len(np.where(signature[:,i[0]]==signature[:,i[1]])[0])/len(signature[:,i[1]]) > 0.5:
            
            u1 = sparse.T.getcol(i[0]).toarray()
            u2 = sparse.T.getcol(i[1]).toarray()
            score = np.logical_and(u1,u2).sum() / float(np.logical_or(u1,u2).sum())
            if score > 0.5:
                pairs.append(sorted((i[0],i[1])))
                np.savetxt('js.txt', sorted(pairs), delimiter=',', fmt='%i')
            else:
                pass
    return sorted(pairs)





def cosine(sparse, signature, candidates):
    pairs = []
    np.seterr(divide='ignore', invalid='ignore')

    for i in candidates:
        if len(np.where(signature[:,i[0]]==signature[:,i[1]])[0])/len(signature[:,i[1]]) > 0.7:
            u1 = sparse[i[0]].toarray()
            u2 = sparse[i[1]].toarray()
            score = dot(u1, u2.T)/(norm(u1)*norm(u2))
            score = score[0][0]
            score = 1 - (math.acos(score)/math.pi)
            if score > 0.67:
                pairs.append(sorted((i[0],i[1])))
                np.savetxt('cs.txt', sorted(pairs), delimiter=',', fmt='%i')
            else:
                pass
        else:
            pass
    
    return sorted(pairs)





def d_cosine(sparse, signature, candidates):
    pairs = []
    np.seterr(divide='ignore', invalid='ignore')

    for i in candidates:
        if len(np.where(signature[:,i[0]]==signature[:,i[1]])[0])/len(signature[:,i[1]]) > 0.7:          
            u1 = sparse[i[0]].toarray()
            u1[u1 > 0] = 1
            u2 = sparse[i[1]].toarray()
            u2[u2 > 0] = 1
            score = dot(u1, u2.T)/(norm(u1)*norm(u2))
            score = score[0][0]
            score = 1 - (math.acos(score)/math.pi)

            if score > 0.67:
                pairs.append(sorted((i[0],i[1])))
                np.savetxt('dcs.txt', sorted(pairs), delimiter=',', fmt='%i')
            else:
                pass
        else:
            pass

    return sorted(pairs)


# Task 1 - jaccard
if args.method == 'js':
    begin = time.time()
    np.random.seed(seed=args.randseed)
    sparse = make_csc(data, method='js')
    print("Done: sparse matrix")
    print("Time: ", time.time()-begin)
    signature = make_signature(sparse, args.randseed, 100, method='js')
    print("Done: signature matrix")
    print("Time: ", time.time()-begin)
    candidates = LSH(signature, b = 20)
    print("Done: found",len(candidates) ,"candidates")
    print("Time: ", time.time()-begin)
    similar = jaccard(sparse, signature, candidates)
    print("Number of pairs:", len(similar), "Pairs:", similar)
    print("Total time elapsed: ", time.time()-begin)

# Task 2 - cosine
if args.method == 'cs':
    begin = time.time()
    np.random.seed(seed=args.randseed)
    sparse = make_csc(data, method='cs')
    print("Done: sparse matrix")
    print("Time: ", time.time()-begin)
    signature = make_signature(sparse, args.randseed, 100, method='cs')
    print("Done: signature matrix")
    print("Time: ", time.time()-begin)
    candidates = LSH(signature, b = 10)
    print("Done: found",len(candidates) ,"candidates")
    print("Time: ", time.time()-begin)
    similar = cosine(sparse, signature, candidates)
    print("Number of pairs:", len(similar), "Pairs:", similar)
    print("Total time elapsed: ", time.time()-begin)
    
    
# Task 3 - discrete cosine
if args.method == 'dcs':
    begin = time.time()
    np.random.seed(seed=args.randseed)
    sparse = make_csc(data, method='dcs')
    print("Done: sparse matrix")
    print("Time: ", time.time()-begin)
    signature = make_signature(sparse, args.randseed, 100, method='dcs')
    print("Done: signature matrix")
    print("Time: ", time.time()-begin)
    candidates = LSH(signature, b = 1)
    print("Done: found",len(candidates) ,"candidates")
    print("Time: ", time.time()-begin)
    similar = d_cosine(sparse,  signature, candidates)
    print("Number of pairs:", len(similar), "Pairs:", similar)
    print("Total time elapsed: ", time.time()-begin)
    