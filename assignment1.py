# Machine Learning Homework 1
# Author: Zack Oldham
# Date: 09/06/2019
# Fit the function: y = 6sin(x+2) + sin(2x+4)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



##############################################################################
# Problem 1

# The true function
def f_true(x):
    y = 6.0 * np.sin(x + 2) + np.sin(2*x + 4)
    return y


# X = float(n, ): univariate data 
# d (int) : degree of plynomial    
def polynomial_transform(X, d):
    array = []
    for x in X:
        row = []
        for i in range(d + 1):
            row.append(x**i)
        array.append(row)
    
    phi = np.matrix(array)
    return phi


# find the weight set for the given Vandermonde matrix and y values
def train_model(Phi, y):
    w = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y.T
    return w


# Phi float(n, d): transformed data
# y float(n, ): labels
# w float(d, ): Linear regression model
# evaluate the model using least mean squared error
# used for both #1 and #2
def evaluate_model(Phi, y, w):
    mse = 0.0

    for i in range(y.size):
        k = y[i] - (Phi[i] @ w.T)
        mse += k @ k

    mse /= y.shape[0]
    
    return float(mse)


################################################################################

# Problem 2
    

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel      
def radial_basis_functions(X, B, gamma=0.1):
    Phi = None
    for x in X:
        row = []
        for b in B:
            row.append(np.e ** (-gamma * ((x - b) ** 2)))
        
        row = [row]
        
        if Phi is None:
            Phi = np.matrix(row)
        else:
            Phi = np.vstack((Phi, row))
    
    return Phi


# Phi float(n, d): transformed data
# y float(n, ): Labels
# Lam float   : regularization parameter
def train_ridge_model(Phi, y, lam):
    w = np.linalg.inv((Phi.T @ Phi) + (lam * np.eye(Phi.shape[0], Phi.shape[1]))) @ Phi.T @ y.T
    return w
    

################################################################################################


def problem1(X_trn, X_val, X_tst, x_true, y_trn, y_val, y_tst, y_true):
    W = {}
    validationErr = {}
    testErr = {}
    """
    1D) I expect choosing d as 15 or 18 to be the best choice 
    because it won't be too huge (over-fitting) or too small (under-fitting)
    """
    for d in range(3, 25, 3):
        # transform training data into d dimensions
        Phi_trn = polynomial_transform(X_trn, d)
        
        # learn model on training data
        W[d] = train_model(Phi_trn, y_trn)
        
        # transform validation data into d dimensions
        Phi_val = polynomial_transform(X_val, d)
        
        # evaluate model on validation data
        validationErr[d] = evaluate_model(Phi_val, y_val, W[d])
        
        # transform test data into d dimensions
        Phi_tst = polynomial_transform(X_tst, d)
        
        # evaluate model on test data
        testErr[d] = evaluate_model(Phi_tst, y_tst, W[d])
        
    # plot all the models
    plt.figure()
    plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
    plt.xlabel("Polynomial Degree", fontsize=16)
    plt.ylabel("Validation/Test Error", fontsize=16)
    plt.xticks(list(validationErr.keys()), fontsize=12)
    plt.legend(["Validation Error", "Test Error"], fontsize=16)
    plt.axis([2, 25, 15, 60])


    # plot the learned models
    plt.figure()
    plt.plot(x_true, y_true, marker=None, linewidth=5, color='k')
    
    for d in range(9, 25, 3):
        X_d = polynomial_transform(x_true, d)
        y_d = X_d @ W[d].T
        plt.plot(x_true, y_d, marker='None', linewidth=2)
        
    plt.legend(['true'] + list(range(9, 25, 3)))
    plt.axis([-8, 8, -15, 15])
    
    
def problem2(X_trn, X_val, X_tst, x_true, y_trn, y_val, y_tst, y_true):
    W = {}
    validationErr = {}
    testErr = {}
    lam = 0.0
    B = X_trn
    
    for i in range(-3, 4):
        
        lam = 10 ** i
        
        # transform training data using radial basis functions
        Phi_trn = radial_basis_functions(X_trn, B)
        
        # learn the model on training data
        W[lam] = train_ridge_model(Phi_trn, y_trn, lam)
        
        # transform validation data using radial basis functions
        Phi_val = radial_basis_functions(X_val, B)
        
        # evaluate model on validation data
        validationErr[lam] = evaluate_model(Phi_val, y_val, W[lam])
        
        # transform test data using radial basis functions
        Phi_tst = radial_basis_functions(X_tst, B)
        
        # evaluate model on test data
        testErr[lam] = evaluate_model(Phi_tst, y_tst, W[lam])
        
    
    # plot lambda vs validationErr and testErr
    plt.figure()
    plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
    plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
    plt.xlabel("Lambda Value", fontsize=16)
    plt.ylabel("Validation/Test Err", fontsize=16)
    plt.xticks(list(validationErr.keys()), fontsize=12)
    plt.legend(["Validation Error", "Test Error"], fontsize=16)
    plt.xscale("log")
    plt.minorticks_off()
    plt.axis([0, 2500, 10, 40])
    
    """
    2C) My results show that a lambda value between 10^-3 and 10^-2 is ideal for this context
    """
    
    
    # Plot the learned models
    plt.figure()
    plt.plot(x_true, y_true, marker=None, linewidth=5, color='k')
    
    
    for i in range(-3, 4):
        lam = 10 ** i
        X_lam = radial_basis_functions(x_true, B)
        y_lam = X_lam @ W[lam].T
        plt.plot(x_true, y_lam, marker='None', linewidth=2)


    plt.legend(["true"] + list(W.keys()))
    plt.axis([-8, 8, -15, 15])
    
    """
    2D) As the value of lambda decreases, the model becomes increasingly accurate. 
    However, when lambda reaches 0, or very close to it, overfitting occurs
    """
        

def main():
    n = 750  # number of data points
    X = np.random.uniform(-7.5, 7.5, n)  # Training examples, in one dimension
    e = np.random.normal(0.0, 5.0, n)  # The error in: y = f(x) + `e`  (Gaussian noise)
    y = f_true(X) + e  # true function values plus error (noise)
    
    plt.figure()
    plt.scatter(X, y, 12, marker='o') # plot the synthetic data
    
    # plot the true function (which is technically unknown i.e. what we want to model)
    x_true = np.arange(-7.5, 7.5, 0.05)
    y_true = f_true(x_true)
    plt.plot(x_true, y_true, marker='None', color='r')
    
    
    # split data points between trainging, validation, and testing
    tst_frac = 0.3
    val_frac = 0.1
    
    # first partition full data set into training set and testing set
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42) 
    
    
    # then partition training set into training set and validation set
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
    
    # plot the three subsets
    plt.figure()
    plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
    plt.scatter(X_val, y_val, 12, marker='o', color='green')
    plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')
    
    y = np.matrix(y)  # convert y to a matrix datatype so that it is compatible with other matrices
    problem1(X_trn, X_val, X_tst, x_true, y_trn, y_val, y_tst, y_true)
    problem2(X_trn, X_val, X_tst, x_true, y_trn, y_val, y_tst, y_true)


if __name__ == "__main__":
    main()




