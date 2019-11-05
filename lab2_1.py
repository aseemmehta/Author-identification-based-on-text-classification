# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:56:51 2019

@author: aseem mehta (am1435)
"""

import re
import math
import pandas as pd
import operator
import copy
from sklearn.model_selection import train_test_split


# Stores the Question based on the current and next value
class Question:

    def __init__(self, column, value, nextValue):
        self.column = column
        self.value = value
        self.nextValue = nextValue

    def match(self, example):
        val = example[self.column]
        return val >= self.value

    def __repr__(self):
        condition = ">="
        v = (self.value + self.nextValue) / 2
        return "Is %s %s %s ?" % (self.column, condition, str(v))


# Creates a leaf node
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


# Creates a Decision node
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# counts the number of label
def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


# Seperates tree into left and right based on question
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


# calculates entropy
def entropy(rows):
    counts = class_counts(rows)
    entropy = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        entropy -= prob_of_lbl * math.log(prob_of_lbl)
    return entropy


# calculates the gain
def info_gain(left, right, Eint):
    p = float(len(left)) / (len(left) + len(right))
    return Eint - p * entropy(left) - (1 - p) * entropy(right)


# Finds the best gain
def find_best_split(rows):
    best_gain = 0
    best_question = None
    Eint = entropy(rows)
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        values = set([row[col] for row in rows])
        values = sorted(values)
        for i in range(len(values) - 1):
            question = Question(col, list(values)[i], list(values)[i + 1])
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, Eint)
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# builds the decision tree
def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


# prints the DT
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = (int(counts[lbl] / total * 100))
    label = max(probs.items(), key=operator.itemgetter(1))[0]
    return label

# calculates number of times feature occurs
def calculateFeature(list, feature):
    countList = []
    for i in list:
        n = 0
        for j in i:
            if feature in j:
                n += 1
        countList.append(n)
    return countList

# creates a table for each book, to be used for classification
def calculateTable(list,author):
    at = calculateFeature(list,"at")
    the = calculateFeature(list,"the")
    quote = calculateFeature(list,'"')
    underscore = calculateFeature(list,"_")
    comma = calculateFeature(list,",")
    stopL = calculateFeature(list,".")
    
    author = [author]*len(comma)
    table = pd.DataFrame({'comma': comma, 'stop': stopL, 'underscore': underscore
                             , 'quote': quote, 'the': the, 'AT': at, 'Author': author})
    return table

# scans a book
def scanBook(file1):
    list = []
    with open(file1) as f:
        word = f.read()
        pw = re.split('\\s', word)
        l = []
        n = 1
        for i in pw:
            if n < 250:
                l.append(i)
                n += 1
            else:
                l.append(i)
                list.append(l)
                l = []
                n = 1
    author = file1[:file1.index(".")]
    return calculateTable(list,author) 

# used to split the data for Decision tree
def test(table):
    values = table.values[:, 0:7]
    train, test = train_test_split(values, test_size=0.3, random_state=100)
    X_train = train[:, 0:6]
    y_train = train[:, 6]
    X_test = test[:, 0:6]
    y_test = test[:, 6]
    return train, test, X_train, X_test, y_train, y_test


# used to split the data for Logistic Classifier
def test1(table):
    values = table[:, 0:7]
    train, test = train_test_split(values, test_size=0.3, random_state=100)
    X_train = train[:, 0:6]
    y_train = train[:, 6]
    X_test = test[:, 0:6]
    y_test = test[:, 6]
    return train, test, X_train, X_test, y_train, y_test


# defines sigmoid function
def sigmoid(y):
    return (1 / (1 + math.exp(-y)))


# returns value between 0 and 1 based on sigmoid
def what(s):
    if s > 0.65:
        return 1
    else:
        return 0


# Calculates the accuracy for logistic classifier for 3 authors
def logisticthree(A):
    C = copy.deepcopy(A)
    D = copy.deepcopy(A)
    E = copy.deepcopy(A)
    author = list(set(A[:,-1]))
    print(author)
    # for ACD
    for i in range(len(A)):
        if author[1] == A[i][-1]:
            C[i][-1] = 1
        else:
            C[i][-1] = 0

    # for HM
    for i in range(len(A)):
        if author[2] == A[i][-1]:
            D[i][-1] = 1
        else:
            D[i][-1] = 0

    # for JA
    for i in range(len(A)):
        if author[0] == A[i][-1]:
            E[i][-1] = 1
        else:
            E[i][-1] = 0

    train1, testing_data1, X_train1, X_test1, y_train1, y_test1 = test1(C)
    w1 = gradient(X_train1, y_train1)
    train2, testing_data2, X_train2, X_test2, y_train2, y_test2 = test1(D)
    w2 = gradient(X_train2, y_train2)
    train, testing_data, X_train, X_test, y_train, y_test = test1(E)
    w3 = gradient(X_train, y_train)

    accuracythree(w1, X_test1, y_test1, w2, X_test2, y_test2, w3, X_test, y_test)


# Calculates the accuracy for logistic classifier for 3 authors
def accuracythree(w1, X_test1, y_test1, w2, X_test2, y_test2, w3, X_test, y_test):
    X1 = X_test1
    X2 = X_test2
    X3 = X_test
    value = []

    for j in range(len(X_test)):
        y1 = w1[-1] + w1[0] * X1[j][0] + w1[1] * X1[j][1] + w1[2] * X1[j][2] + w1[1] * X1[3][3] + w1[4] * X1[j][4]
        y2 = w2[-1] + w2[0] * X2[j][0] + w2[1] * X2[j][1] + w2[2] * X2[j][2] + w2[1] * X2[3][3] + w2[4] * X2[j][4]
        y3 = w3[-1] + w3[0] * X3[j][0] + w3[1] * X3[j][1] + w3[2] * X3[j][2] + w3[1] * X3[3][3] + w3[4] * X3[j][4]
        y = max(y1, y2, y3)
        s = sigmoid(y)
        val = what(s)
        value.append(val)

    n = 0
    for i in range(len(X_test)):
        if y_test[i] == value[i]:
            n += 1
    print(n / len(X_test))


# Calculates the accuracy for logistic classifier for 2 authors
def logistic(A):
    C = copy.deepcopy(A)
    author = list(set(A[:,-1]))
    for i in range(len(A)):
        if author[0] == A[i][-1]:
            C[i][-1] = 1
        else:
            C[i][-1] = 0

    train, testing_data, X_train, X_test, y_train, y_test = test1(C)

    w = gradient(X_train, y_train)
    accuracy(w, X_test, y_test)


# Used to calculate gradient descent
def delw0(a, b):
    return (a - b)


# Used to calculate gradient descent
def delw1(a, b, c):
    return (a - b) * c


# Calculates the accuracy for logistic classifier for 2 authors
def accuracy(w, X_test, y_test):
    value = []
    X = X_test
    for j in range(len(X_test)):
        y2 = w[-1] + w[0] * X[j][0] + w[1] * X[j][1] + w[2] * X[j][2] + w[1] * X[3][3] + w[4] * X[j][4]
        s = sigmoid(y2)
        val = what(s)
        value.append(val)

    n = 0
    for i in range(len(X_test)):
        if y_test[i] == value[i]:
            n += 1
    print(n / len(X_test))


# calculates gradient descent
def gradient(X_train, y_train):
    X = X_train
    y = y_train
    iteration = int(input('enter number of iterations\n'))
    # iteration = 2000
    rate = 10 ** -6
    lenofdata = len(X_train)
    colofdata = len(X_train[0])
    w = [1] * (colofdata)
    for i in range(iteration):

        if (i % 10000 == 0):
            print("% of Data trained: ", i / iteration * 100)

        grad = [0] * colofdata
        for j in range(lenofdata):
            y1 = w[-1] + w[0] * X[j][0] + w[1] * X[j][1] + w[2] * X[j][2] + w[1] * X[3][3] + w[4] * X[j][4]

            grad[0] += delw0(y[j], y1)
            grad[1] += delw1(y[j], y1, X[j][0])
            grad[2] += delw1(y[j], y1, X[j][1])
            grad[3] += delw1(y[j], y1, X[j][2])
            grad[4] += delw1(y[j], y1, X[j][3])
            grad[5] += delw1(y[j], y1, X[j][4])
        w[-1] += rate * grad[0]
        w[0] += rate * grad[1]
        w[1] += rate * grad[2]
        w[2] += rate * grad[3]
        w[3] += rate * grad[4]
        w[4] += rate * grad[5]
    return (w)


# builds the decision tree
def decision_tree(train, testing_data):
    my_tree = build_tree(train)

    model_print = input('Print Decision tree model Yes/No \n')
    if model_print == "Yes":
        print_tree(my_tree)
    check = []
    for row in testing_data:
        check.append(print_leaf(classify(row, my_tree)))
        print("Actual: %s. Predicted: %s" % (row[-1], print_leaf(classify(row, my_tree))))

    n = 0
    for i in range(len(check)):
        if testing_data[i][-1] == check[i]:
            n += 1
    print(n / len(check))


def main():
    #text1 = input("Name of first text file \n")
    #text2 = input("Name of second text file \n")
    #text3 = input("Name of third text file \n")
    text1 = "ACD.txt"
    text2 = "HM.txt"
    text3 = "JA.txt"
    tableA = scanBook(text1)
    tableH = scanBook(text2)
    tablej = scanBook(text3)
    table = tableA.append(tableH, ignore_index=True)
    table1 = table.append(tablej, ignore_index=True)
    A = table.values[:, 0:7]
    A1 = table1.values[:, 0:7]

    Act = input('Enter 1,2,3: Accuracy to be predicted by 1. Decion tree, 2. logistic Classifier, 3.exit \n')

    while (Act != "3"):
        if Act == "1":
            train, testing_data, X_train, X_test, y_train, y_test = test(table)
            Auth = input("run with 2 or 3 authors \n")
            if Auth == "3":
                train, testing_data, X_train, X_test, y_train, y_test = test(table1)
            decision_tree(train, testing_data)
            Act = input('Enter 1,2,3: Accuracy to be predicted by 1. Decion tree, 2. logistic Classifier, 3.exit \n')
        else:
            Auth = input("run with 2 or 3 authors \n")
            if Auth == "3":
                logisticthree(A1)
            else:
                logistic(A)
            Act = input('Enter 1,2,3: Accuracy to be predicted by 1. Decion tree, 2. logistic Classifier, 3.exit \n')


if __name__ == '__main__':
    main()
