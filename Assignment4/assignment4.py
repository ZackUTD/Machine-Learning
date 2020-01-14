#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Zack Oldham
Netid: zoo150030
CS 6375.002
Assignment 4
SVMs/K-NN
"""

import numpy as np
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC as svc
from sklearn.neighbors import KNeighborsClassifier as knc



################################################################################
# Part 1
################################################################################


"""
DO NOT EDIT THIS FUNCTION -- MAKE A COPY IF YOU WANT TO PLAY AROUND
"""
def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    # Generate a non-linear data set
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30)  # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
    y[ind] = 1 - y[ind]

    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac,
    random_state=42)

    # Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac,
    random_state=42)

    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)



"""
DO NOT EDIT THIS FUNCTION -- MAKE A COPY IF YOU WANT TO PLAY AROUND
"""
def visualize(models, param, X, y):
    # Initialize plotting
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])

    # Create a mesh
    xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
    yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),
                             np.arange(yMin, yMax, 0.01))

    for i, (p, clf) in enumerate(models.items()):
        # if i > 0:
        #   break
        r, c = np.divmod(i, 3)
        ax = axes[r, c]

        # Plot contours
        zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

        if (param == 'C' and p > 0.0) or (param == 'gamma'):
            ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

        # Plot data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('{0} = {1}'.format(param, p))


# Get the error percentage for a given classifier on a given dataset
def getError(X, y, clf):
    errCount = 0

    yPred = clf.predict(X)

    for (yt, yp) in zip(y, yPred):
        if yt != yp:
            errCount += 1

    return errCount/len(X)


def part1a(X_trn, y_trn, X_val, y_val, X_tst, y_tst):

    # Learn support vector classifiers with a radial-basis function kernel with
    # fixed gamma = 1 / (n_features * X.std()) and different values of C
    C_range = np.arange(-3.0, 6.0, 1.0)
    C_values = np.power(10.0, C_range)

    models = dict()
    trnErr = dict()
    valErr = dict()

    # generate models for C value range
    for C in C_values:
        models[C] = svc(C=C, gamma='scale')
        models[C].fit(X_trn, y_trn)
        trnErr[C] = getError(X_trn, y_trn, models[C])*100
        valErr[C] = getError(X_val, y_val, models[C])*100


    visualize(models, 'C', X_trn, y_trn)

    # Calculate model errors and graph
    plt.figure()
    plt.title("Changing C values vs Train/Validation Error")
    plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(valErr.keys()), list(valErr.values()), marker='s', linewidth=3, markersize=12)
    plt.xlabel("C Value", fontsize=16)
    plt.ylabel("Train/Validation Error %", fontsize=16)
    plt.xticks(list(trnErr.keys()), fontsize=12)
    plt.legend(["Train Error", "Validation Error"], fontsize=16)
    plt.xscale("log")
    plt.minorticks_off()
    plt.axis([0.0001, 1000000, 0, 100])



    bestAcc = 0
    bestC = 0

    for C in models:
        curAcc = 1 - getError(X_tst, y_tst, models[C])
        if curAcc > bestAcc:
            bestAcc = curAcc
            bestC = C


    print("Maximum SVM accuracy of {0:4.2f}%".format(bestAcc*100), "achieved with C =", bestC)




def part1b(X_trn, y_trn, X_val, y_val, X_tst, y_tst):
    gamma_range = np.arange(-2.0, 4.0, 1.0)
    gamma_values = np.power(10.0, gamma_range)

    models = dict()
    trnErr = dict()
    valErr = dict()

    for G in gamma_values:
        models[G] = svc(C=10, gamma=G)
        models[G].fit(X_trn, y_trn)
        trnErr[G] = getError(X_trn, y_trn, models[G])*100
        valErr[G] = getError(X_val, y_val, models[G])*100


    visualize(models, 'gamma', X_trn, y_trn)



    # Calculate model errors and graph
    plt.figure()
    plt.title("Changing Gamma values vs Train/Validation Error")
    plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(valErr.keys()), list(valErr.values()), marker='s', linewidth=3, markersize=12)
    plt.xlabel("Gamma Value", fontsize=16)
    plt.ylabel("Train/Validation Error %", fontsize=16)
    plt.xticks(list(trnErr.keys()), fontsize=12)
    plt.legend(["Train Error", "Validation Error"], fontsize=16)
    plt.xscale("log")
    plt.minorticks_off()
    plt.axis([0.0001, 1000000, 0, 50])


    # Find best model
    bestAcc = 0
    bestG = 0

    for G in models:
        curAcc = 1 - getError(X_tst, y_tst, models[G])
        if curAcc > bestAcc:
            bestAcc = curAcc
            bestG = G


    print("Maximum SVM accuracy of {0:4.2f}%".format(bestAcc*100), "achieved with gamma =", bestG)



################################################################################
# Part 2
################################################################################


# display a table of the given data with the given title
def tabulate(data, title):
    rowlabels = []
    columnlabels = []

    for val in list(data.keys()):
        rowlabels.append("C = " + str(val))

    for val in list(data[0.1].keys()):
        columnlabels.append("gamma = " + str(val))


    dataAsList = []

    for c in data:
        dataAsList.append(["{0:4.2f}".format(data[c][g]) for g in data[c]])



    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title, fontsize=15)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.axis("off")
    ax.table(cellText=dataAsList, rowLabels=rowlabels, colLabels=columnlabels, loc="center")




def part2(X_trn, y_trn, X_val, y_val, X_tst, y_tst):
    C_range = np.arange(-2.0, 5.0 , 1.0)
    C_values = np.power(10.0, C_range)

    gamma_range = np.arange(-3.0, 3.0, 1.0)
    gamma_values = np.power(10.0, gamma_range)

    models = {}
    trnErr = {}
    valErr = {}

    maxAcc = 0
    bestC = None
    bestG = None

    for c in C_values:
        models[c] = {}
        trnErr[c] = {}
        valErr[c] = {}
        for g in gamma_values:
            models[c][g] = svc(C=c, gamma=g)
            models[c][g].fit(X_trn, y_trn)
            trnErr[c][g] = getError(X_trn, y_trn, models[c][g])*100
            valErr[c][g] = getError(X_val, y_val, models[c][g])*100

            curAcc = 100 - valErr[c][g]
            if curAcc > maxAcc:
                maxAcc = curAcc
                bestC = c
                bestG = g



    # Table of error values for train and validation datasets
    tabulate(trnErr, "Training Error")
    tabulate(valErr, "Validation Error")


    bestAccuracy = 100 - getError(X_tst, y_tst, models[bestC][bestG])*100

    print("Maximum SVM accuracy of {0:4.2f}% on breast cancer data achieved with".format(bestAccuracy),
          "C = {} and gamma = {}".format(bestC, bestG))




################################################################################
# Part 3
################################################################################


def part3(X_trn, y_trn, X_val, y_val, X_tst, y_tst):
    N = [1, 5, 11, 15, 21]

    models = {}
    trnErr = {}
    valErr = {}

    for n in N:
        models[n] = knc(n_neighbors=n, algorithm="kd_tree")
        models[n].fit(X_trn, y_trn)
        trnErr[n] = getError(X_trn, y_trn, models[n])*100
        valErr[n] = getError(X_val, y_val, models[n])*100


    # Calculate model errors and graph
    plt.figure()
    plt.title("Breast Cancer Classification Error with KNN")
    plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(valErr.keys()), list(valErr.values()), marker='s', linewidth=3, markersize=12)
    plt.xlabel("Number of Neighbors", fontsize=16)
    plt.ylabel("Train/Validation Error %", fontsize=16)
    plt.xticks(list(trnErr.keys()), fontsize=12)
    plt.legend(["Train Error", "Validation Error"], fontsize=16)
    #plt.minorticks_off()
    plt.axis([0, 25, 0, 10])



################################################################################
# Driver
################################################################################


# Collect the data from the input files
def getData():
    trnData = np.loadtxt("./wdbc_trn.csv", delimiter=',')
    valData = np.loadtxt("./wdbc_val.csv", delimiter=',')
    tstData = np.loadtxt("./wdbc_tst.csv", delimiter=',')

    X_trn = trnData[:, 1:]
    y_trn = trnData[:, 0]

    X_val = valData[:, 1:]
    y_val = valData[:, 0]

    X_tst = tstData[:, 1:]
    y_tst = tstData[:, 0]


    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)



def main():

    print("The scikit-learn version is {}".format(sklearn.__version__))

    n_samples = 300

    (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

    part1a(X_trn, y_trn, X_val, y_val, X_tst, y_tst)
    part1b(X_trn, y_trn, X_val, y_val, X_tst, y_tst)

    (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = getData()
    part2(X_trn, y_trn, X_val, y_val, X_tst, y_tst)
    part3(X_trn, y_trn, X_val, y_val, X_tst, y_tst)

if __name__ == "__main__":
    main()



