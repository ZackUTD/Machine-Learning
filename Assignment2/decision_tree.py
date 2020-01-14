# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
# Anjum Chida (anjum.chida@utdallas.edu)
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt

#sklearn is used exclusively for parts c/d/e as noted in instructions but will be commented out for submission
#from sklearn.metrics import confusion_matrix as cm
#from sklearn.tree import DecisionTreeClassifier as dtc
#from sklearn.tree import export_graphviz as egz
#from sklearn.model_selection import train_test_split as tts

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    partition = {}
    
    for i in len(x):
        partition[x[i]].append(i);
        


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    
    y_vals = {}
    
    for l in y:
        if l not in y_vals:
            y_vals[l] = 0
            
        y_vals[l] += 1
    
    ent = 0
    
    for val in y_vals:
        pv = y_vals[val]/len(y)
        ent += pv * np.log2(pv)
    
    
    #equations in slides indicate that entropy should be NEGATED value of above sum
    ent *= -1
    return ent



def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    
    subs = {}
    counts = {}
    
    
    for v,l in zip(x, y):
        if v not in counts:
            counts[v] = {}
            subs[v] = []
        if l not in counts[v]:
            counts[v][l] = 0
        
        counts[v][l] += 1
        subs[v].append(l)
    
    Hyx = 0
    
    for v in subs:
        wAvg = 0
        for l in counts[v]: 
            wAvg += counts[v][l]
        wAvg /= len(x)
        Hyx += wAvg * entropy(subs[v])
    
    return entropy(y) - Hyx
    


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    #if we haven't recursed yet then initialize the set of attribute/value pairs    
    if depth == 0:
        attribute_value_pairs = []
    
        for inst in x:
            for i in range(len(inst)):
                if (i, inst[i]) not in attribute_value_pairs:
                    attribute_value_pairs.append((i, inst[i]))
       
    
    pure = True
    
    for l in y:
        if l != y[0]:
            pure = False
            break
    
    if pure:
        return y[0]
        
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        
        counts = {}
        
        for l in y:
            if l not in counts:
                counts[l] = 0

            counts[l] += 1
            
        maxcount = max(counts.values())
        max_idx = list(counts.values()).index(maxcount)
        return list(counts.keys())[max_idx]
    
    
    tree = {}
    
    MutInfo = []
    xsub = []
    
    for a, v in attribute_value_pairs:
        for inst in x:
            if inst[a] == v:
                xsub.append(inst[a])
            else:
                xsub.append("")
        
        MutInfo.append(mutual_information(xsub, y))
        
    bestPair_idx = MutInfo.index(max(MutInfo))
    bestPair = attribute_value_pairs[bestPair_idx]
    attribute_value_pairs.remove(bestPair)
    
    x_true = []
    x_false = []
    y_true = []
    y_false = []
    a, v = bestPair
    
    for inst, l in zip(x, y):
        if inst[a] == v:
            x_true.append(inst)
            y_true.append(l)
        else:
            x_false.append(inst)
            y_false.append(l)
    
    depth += 1
    
    if len(y_false) != 0:
        tree[bestPair + (False,)] = id3(x_false, y_false, attribute_value_pairs, depth, max_depth)
    
    if len(y_true) != 0:
        tree[bestPair + (True,)] = id3(x_true, y_true, attribute_value_pairs, depth, max_depth)
        
    return tree
    
            
            
def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    
    if type(tree) != dict:
        return tree
    
    for node in tree:
        a, v, r = node
        
        if (r and x[a] == v) or (not r and x[a] != v):
            return predict_example(x, tree[node])
        
    


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    
    errs = 0
    
    for yp, yt in zip(y_pred, y_true):
        if yp != yt:
            errs += 1
    
    
    return errs/len(y_true)


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

"""!!!!!NOTE: I will need to modify the path to graphviz so that I can show an img for the report, BUT
I MUST return the path to its original value before submitting the assignment as the autograder probably needs to
use the path that is given!!!!!!"""
def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


# pre-process a column into numerical values
def clean(X):
    X_clean = []
    av_pairs = {}
    
    code = 0
    
    for x in X:
        if x not in av_pairs:
            av_pairs[x] = code
            code += 1
        
        X_clean.append(av_pairs[x])
    
    
    return X_clean


# pre-process all of the data in a set into numerical values
def preprocess(data):
    clean_data = []
    
    clean_cols = []
    
    for i in range(len(data[0])):
        clean_cols.append(clean(data[:, i]))
        
    for j in range(len(data)):
        row = []
        
        for col in clean_cols:
            row.append(col[j])
        
        clean_data.append(row)
        
    return np.array(clean_data)
        

if __name__ == '__main__':
    # Load the training data for monks-1
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn_m1 = M[:, 0]
    Xtrn_m1 = M[:, 1:]

    # Load the test data for monks-1
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst_m1 = M[:, 0]
    Xtst_m1 = M[:, 1:]
    
    # Load the training data for monks-2
    M = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn_m2 = M[:, 0]
    Xtrn_m2 = M[:, 1:]

    # Load the test data for monks-2
    M = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst_m2 = M[:, 0]
    Xtst_m2 = M[:, 1:]
    
    # Load the training data for monks-3
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn_m3 = M[:, 0]
    Xtrn_m3 = M[:, 1:]

    # Load the test data for monks-3
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst_m3 = M[:, 0]
    Xtst_m3 = M[:, 1:]
    
    

    
    m1_trees = {}  # we'll need the trees of depth 1, 3, 5 of monks-1 for part (c) later
    
    y_pred = {"m1": {"trn": [], "tst": []},
              "m2": {"trn": [], "tst": []},
              "m3": {"trn": [], "tst": []}
              }

    # Learn the trees for each problem for each depth, and predict on trn and tst 
    # data through each tree to populate y_pred lists
    
    for d in range(10):
        m1 = id3(Xtrn_m1, ytrn_m1, max_depth=d+1)
        m2 = id3(Xtrn_m2, ytrn_m2, max_depth=d+1)
        m3 = id3(Xtrn_m3, ytrn_m3, max_depth=d+1)        
        
        y_pred["m1"]["trn"].append([predict_example(x, m1) for x in Xtrn_m1])
        y_pred["m1"]["tst"].append([predict_example(x, m1) for x in Xtst_m1])
        
        y_pred["m2"]["trn"].append([predict_example(x, m2) for x in Xtrn_m2])
        y_pred["m2"]["tst"].append([predict_example(x, m2) for x in Xtst_m2])
        
        y_pred["m3"]["trn"].append([predict_example(x, m3) for x in Xtrn_m3])
        y_pred["m3"]["tst"].append([predict_example(x, m3) for x in Xtst_m3])
        
        if d+1 == 1 or d+1 == 3 or d+1 == 5:
            m1_trees[d+1] = m1
        
     
        
    errs = {"m1": {"trn": [], "tst": []},
            "m2": {"trn": [], "tst": []},
            "m3": {"trn": [], "tst": []}
            }
    
    # Compute errpr for trn and tst data for each problem
    for d in range(10):
        errs["m1"]["trn"].append(float("{0:4.2f}".format(compute_error(ytrn_m1, y_pred["m1"]["trn"][d]) * 100)))
        errs["m1"]["tst"].append(float("{0:4.2f}".format(compute_error(ytst_m1, y_pred["m1"]["tst"][d]) * 100)))
        
        errs["m2"]["trn"].append(float("{0:4.2f}".format(compute_error(ytrn_m2, y_pred["m2"]["trn"][d]) * 100)))
        errs["m2"]["tst"].append(float("{0:4.2f}".format(compute_error(ytst_m2, y_pred["m2"]["tst"][d]) * 100)))
        
        errs["m3"]["trn"].append(float("{0:4.2f}".format(compute_error(ytrn_m3, y_pred["m3"]["trn"][d]) * 100)))
        errs["m3"]["tst"].append(float("{0:4.2f}".format(compute_error(ytst_m3, y_pred["m3"]["tst"][d]) * 100)))
        
        
    
    # Part (b): plot depth vs train & test error for each problem (monks 1-3)
    
    print(errs["m1"]["trn"])
    print(errs["m1"]["tst"])
    
    # Monks-1
    x_vals = np.arange(1, len(y_pred["m1"]["trn"]) + 1, 1)    
    plt.figure()
    plt.title("Monks-1 Error")
    plt.plot(x_vals, errs["m1"]["trn"], marker='s', color='blue', markersize=8)
    plt.plot(x_vals, errs["m1"]["tst"], marker='o', color='orange', markersize=8)
    plt.legend(["Training", "Testing"])
    plt.xlabel("Tree Depth", fontsize=16)
    plt.ylabel("% Error", fontsize=16)
    plt.axis([1, max(x_vals), 0, 100])

    
    # Monks-2
    x_vals = np.arange(1, len(y_pred["m1"]["trn"]) + 1, 1)    
    plt.figure()
    plt.title("Monks-2 Error")
    plt.plot(x_vals, errs["m2"]["trn"], marker='s', color='blue', markersize=8)
    plt.plot(x_vals, errs["m2"]["tst"], marker='o', color='orange', markersize=8)
    plt.legend(["Training", "Testing"])
    plt.xlabel("Tree Depth", fontsize=16)
    plt.ylabel("% Error", fontsize=16)
    plt.axis([1, max(x_vals), 0, 100])


    # Monks-3
    x_vals = np.arange(1, len(y_pred["m1"]["trn"]) + 1, 1)    
    plt.figure()
    plt.title("Monks-3 Error")    
    plt.plot(x_vals, errs["m3"]["trn"], marker='s', color='blue', markersize=8)
    plt.plot(x_vals, errs["m3"]["tst"], marker='o', color='orange', markersize=8)
    plt.legend(["Training", "Testing"])
    plt.xlabel("Tree Depth", fontsize=16)
    plt.ylabel("% Error", fontsize=16)
    plt.axis([1, max(x_vals), 0, 100])
    
    
    
    # Part (c): For monks-1, visualize the trees and confusion matrices for depths 1, 3, 5
    print("confusion matrices using id3 func:")
    
    # Following section is commented out to prevent autograder issues as it is uses sklearn methods.
    # (only used as specified by instructions and where necessary for those instructions)
    
    """
    for t in m1_trees:
        dot_str = to_graphviz(m1_trees[t])
        render_dot_file(dot_str, './m1_d' + str(t) + '_tree')
        print("depth = {}:".format(t))
        print(cm(ytst_m1, y_pred["m1"]["tst"][t]))
        
    
    # Part(d): For monks-1 use sklearn's DecisionTreeClassifier to learn a decision tree
    # using entropy as criterion for depth 1, 3, 5. Show the confusion matrix.
    
    print("confusion matrices using sklearn:")
    
    
    for d in range(1, 6, 2):
        t = dtc(criterion='entropy', max_depth=d)
        t.fit(Xtrn_m1, ytrn_m1)
        sk_pred = t.predict(Xtst_m1)
        print("depth = {}".format(d))
        print(cm(ytst_m1, sk_pred))
        
        dot_str = egz(t)
        render_dot_file(dot_str, './sklm1_d' + str(d) + '_tree')
    
       
    # Part(e): Repeat steps (c) & (d) using your own data set from 
    # UCI repository and report confusion matrices
    
    #Load data for cars
    M = np.genfromtxt('./car.data', missing_values=0, skip_header=0, delimiter=',', dtype=str)
    M = preprocess(M)
    y = M[:, 6]
    X = M[:,:6]
    
    Xtrn_c, Xtst_c, ytrn_c, ytst_c = tts(X, y, test_size=0.6, random_state=42)
    
    
    # repeat part (c) on cars data
    
    print("confusion matrices of cars data using id3 func:")
    
    for d in range(1, 6, 2):
        t = id3(Xtrn_c, ytrn_c, max_depth=d)
        yc_pred = [predict_example(x, t) for x in Xtst_c]
        print("depth = {}:".format(d))
        print(cm(ytst_c, yc_pred))
        
        dot_str = to_graphviz(t)
        render_dot_file(dot_str, './cars_d' + str(d) + '_tree')
    
    
    print("confusion matrices of cars using sklearn:")
    
    
    # repeat part (d) on cars data    
    for d in range(1, 6, 2):
        t = dtc(criterion='entropy', max_depth=d)
        t.fit(Xtrn_c, ytrn_c)
        sk_pred = t.predict(Xtst_c)
        print("depth = {}".format(d))
        print(cm(ytst_c, sk_pred))
        
        dot_str = egz(t)
        render_dot_file(dot_str, './sklcars_d' + str(d) + '_tree')
    """
