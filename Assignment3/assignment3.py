#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zack Oldham
Netid: zoo150030
10/20/2019
Assignment 3 - Naive Bayes Document Classification
"""

import numpy as np
import re
import os
from collections import Counter


################################################################################
# Collect/Pre-Process Data
################################################################################

# Given a directory Dir, preprocess each file in the directory
# into a list of the words it contains.
# Return a list containing all of the file word lists
def getDocs(Dir):
    dirContents = []
    dir_enc = os.fsencode(Dir)

    for file in os.listdir(dir_enc):
        fname = os.fsdecode(file)
        if fname.endswith(".txt"):
            dirContents.append(preProcess(Dir + fname))

    return dirContents


# Given a filename, read it contents (delimiting on any numeric or special character),
# and return words in a list.
def preProcess(fileName):
    cleanFile = []
    DELIM = (" ", "\t", "\n", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "~", "`", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "\\", "]", "}", "[", "{", ";", ":", "/", ",", "?", ".", ">", "<")

    file = open(fileName, encoding="latin-1")
    raw = file.read()

    regexPattern = '|'.join(map(re.escape, DELIM))

    for token in re.split(regexPattern, raw):
        if not token:
            continue

        cleanFile.append(token)

    return cleanFile


# Given a set of documents D, determine the vocabulary, V
def extractVocab(D):
    # Get the set of all words we have seen over both classes of documents

    V = None

    for doc in D:
        for token in doc:
            if V is None:
                V = {None}
            else:
                V.add(token)

    return V
    
    

# get the class labels for the given number of ham/spam documents (1 for ham, 0 for spam)
def getClassLabels(numHam,numSpam):
    Yvals = []
    for i in range(numHam):
        Yvals.append(1)

    for i in range(numSpam):
        Yvals.append(0)
    
    return Yvals
    

# Remove all of the stopwords from a given set of documents (stopwords should be located in a "\n" delimited file called stopwords.txt)
def removeStopWords(Docs):
    file = open("./stopwords.txt")
    stops = file.read()
    
    for i in range(len(Docs)):
        for token in Docs[i]:
            if token in stops:
                Docs[i].remove(token)
                
    
    return Docs 
    

################################################################################
# Train NB Classifier
################################################################################

# Given a document set D, count the total number of words in the set
def wordCount(D):
    count = 0

    for doc in D:
        count += len(doc)

    return count


# Given a document set D, count the number of times each token appears throughout the set (adding one for Laplace smoothing)
# If any token from the vocab, V, is not found in this document set, set its count to 1 for Laplace smoothing
def tokenCounts(D, V):
    tokCounts = {}

    for doc in D:
        for token in doc:
            if token not in tokCounts:
                tokCounts[token] = 1

            tokCounts[token] += 1

    for token in V:
        if token not in tokCounts:
            tokCounts[token] = 1

    return tokCounts


# Train a Naive Bayes classifier on a set of training documents
def trainNB(hamDocs, spamDocs):
    Docs = hamDocs + spamDocs
    Vocab = extractVocab(Docs)
    ham_wdCt = wordCount(hamDocs)
    spam_wdCt = wordCount(spamDocs)

    tokCounts_ham = tokenCounts(hamDocs, Vocab)
    tokCounts_spam = tokenCounts(spamDocs, Vocab)

    hamPrior = 0
    spamPrior = 0
    conProbs_ham = {}
    conProbs_spam = {}


    # compute prior P(ham)
    hamPrior = len(hamDocs)/len(Docs)

    # compute prior P(spam)
    spamPrior = len(spamDocs)/len(Docs)


    # Compute conditional probabilities for tokens given ham class P(token|ham)
    for token in Vocab:
        conProbs_ham[token] = tokCounts_ham[token]/(ham_wdCt + len(Vocab))


    # Compute conditional probabilities for tokens given spam class P(token|spam)
    for token in Vocab:
        conProbs_spam[token] = tokCounts_spam[token]/(spam_wdCt + len(Vocab))


    return (hamPrior, spamPrior, conProbs_ham, conProbs_spam, Vocab)


################################################################################
# Test NB Classifier
################################################################################

# For a given document, extract all of the tokens from it that are in the vocabulary, V.
def extractTokens(doc, V):
    tokens = []

    for token in doc:
        if token in V:
            tokens.append(token)

    return tokens


# For a given document, determine whether it is a spam document or ham document.
def classifyDocument(doc, stats):
    hamPrior, spamPrior, conProbs_ham, conProbs_spam, Vocab = stats
    
    W = extractTokens(doc, Vocab)

    Prob_ham = np.log2(hamPrior)
    Prob_spam = np.log2(spamPrior)

    for token in W:
        Prob_ham += np.log2(conProbs_ham[token])
        Prob_spam += np.log2(conProbs_spam[token])

    if Prob_ham >= Prob_spam:
        return 1
    else:
        return 0

# Compute the accuracy of a NB classifier on a set of test data
def NBaccuracy(ham_Tst, spam_Tst, stats):
    correctCount = 0
       
    for doc in ham_Tst:
        if classifyDocument(doc, stats) == 1:
           correctCount += 1

    for doc in spam_Tst:
        if classifyDocument(doc, stats) == 0:
            correctCount += 1
            
    NB_accuracy = correctCount/len(ham_Tst + spam_Tst)

    print("Naive Bayes Accuracy: {0:4.2f}%".format(NB_accuracy * 100))



################################################################################
# Train LogReg Classifier
################################################################################


# For a document, doc, and a vocabulary, V, create a vector whose indices
# contain the frequency of each vocab word in the doc.
def getFreqs(doc, V):
    freqs = [1]

    counts = Counter(doc)

    for token in V:
        if token in counts:
            freqs.append(counts[token]/len(doc))
        else:
            freqs.append(0)

    return freqs


# For a document set D and vocabulary Vocab, form a list of the attribute vectors for each document.
def getVectors(D, Vocab):
    trnSet = []

    for doc in D:
        trnSet.append(getFreqs(doc, Vocab))

    return trnSet

# initialize the vector of weights for logistic regression to all zeroes
def initWeights(size):
    temp = np.zeros((1, size))
    weights = np.ndarray.tolist(temp)[0]
    return weights

# Determine P(class=1 | X, W) where X is the attribute vector for a document,
# and W is the learned weight vector.
def logRegEst(X, W):
    prob0 = 1 / (1 + np.exp(W[0] + np.dot(X[1:], W[1:])))

    return float(1 - prob0)


# Update a set of weights, W, using the formula given in the newest Mitchell chapter.
# (Where X is the set of training examples, Y is the corresponding class labels,
# lam is the selected lambda value, and n is the gradient ascent factor).
def updateW(W, X, Y, lam, n):
    for i in range(len(W)):
        for (x, y) in zip(X, Y):
            if x[i] > 0:
                Z = logRegEst(x, W)
                W[i] += (x[i] * (y - Z))
                W[i] *= n

            W[i] -= n*lam*W[i]
    return W


# Train a logistic regression classifier using a document set D, and their class labels Y.
# In doing this perform gradient ascent until a specified cutoff point, cut, using parameters lam and n
# for the lambda value and gradient ascent factor, respectively.
def trainLogReg(D, Y, V, n, lam, cut):
    X = getVectors(D, V)  # The set of attribute vectors that represent each document
    W = initWeights(len(V) + 1)

    for i in range(cut):
        W = updateW(W, X, Y, lam, n)
        

    return W


################################################################################
# Test LogReg Classifier
################################################################################


# test the learned logistic regression classifier on a given document.
def testLogReg(weights, Docs, Vocab):

    predClasses = []
    trnExamples = getVectors(Docs, Vocab)

    for ex in trnExamples:
        if logRegEst(ex, weights) >= 0.5:
            predClasses.append(1)
        else:
            predClasses.append(0)

    return predClasses
    
    
# Compute the accuracy of an LR classifier on a test set
def LRaccuracy(Docs, Ytst, weights, Vocab):
    
    Ypred = testLogReg(weights, Docs, Vocab)
    correctCount = 0
    
    for (yp, yt) in zip(Ypred, Ytst):
        if yp == yt:
            correctCount += 1
    
    accuracy = correctCount/len(Ytst) * 100
    
    print("Logistic Regression Accuracy: {0:4.2f}%".format(accuracy))
    

################################################################################
# Driver
################################################################################
def main():
    ham_Trn = getDocs("./train/ham/")
    spam_Trn = getDocs("./train/spam/")

    ham_Tst = getDocs("./test/ham/")
    spam_Tst = getDocs("./test/spam/")
    
    
    hamTrn_nostops = removeStopWords(ham_Trn)
    spamTrn_nostops = removeStopWords(spam_Trn)
    
    hamTst_nostops = removeStopWords(ham_Tst)
    spamTst_nostops = removeStopWords(spam_Tst)
    
    Y_trn = getClassLabels(len(ham_Trn), len(spam_Trn))
    Y_tst = getClassLabels(len(ham_Tst), len(spam_Tst))

    stats_stops = trainNB(ham_Trn, spam_Trn)
    stats_nostops = trainNB(hamTrn_nostops, spamTrn_nostops)
    Vocab_stops = stats_stops[4]
    Vocab_nostops = stats_nostops[4]
    
    W_stops = trainLogReg(ham_Trn + spam_Trn, Y_trn, Vocab_stops, n=0.01, lam=0.01, cut=10)
    W_nostops = trainLogReg(hamTrn_nostops + spamTrn_nostops, Y_trn, Vocab_nostops, n=0.01, lam=0.01, cut=10)
    
    
    print("With Stop Words:")
    NBaccuracy(ham_Tst, spam_Tst, stats_stops)
    LRaccuracy(ham_Tst + spam_Tst, Y_tst, W_stops, Vocab_stops)
    
    print("Without Stop Words:")
    NBaccuracy(ham_Tst, spam_Tst, stats_nostops)
    LRaccuracy(hamTst_nostops + spamTst_nostops, Y_tst, W_nostops, Vocab_nostops)

if __name__ == "__main__":
    main()




