# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:21:45 2019

@author: nlove
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def scale(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

chess = pd.read_csv('krkopt.data',
                      names=['White King file', 'White King rank', 'White Rook file', 'White Rook rank', 'Black King file', 'Black King rank','depth-of-win'])
cleanup_chess = {"a": 1,"b": 2,"c": 3,"d": 4,"e": 5,"f": 6,"g": 7,"h": 8}
chess.replace(cleanup_chess, inplace=True)
#chess = normalize(chess)
carr = np.array(chess)
c_data, c_tar = carr[:, :-1], carr[:, -1]

chX_train, chX_test, chy_train, chy_test = train_test_split(
        c_data, c_tar, test_size = .3)

mush = pd.read_csv('agaricus-lepiota.data', names = ['ediblevspoison','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment','gill-spacing','gill-size','gill-color',
                                                     'stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring',
                                                     'veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])
mush = mush.drop(['stalk-root'], axis=1)
#mush = mush.drop(['bruises'], axis=1)
mush = pd.get_dummies(mush)
musharr = np.array(mush)
mush_data, mush_tar = musharr[:, 1:], musharr[:, 0]
names = np.unique([str(i) for i in mush])
names = np.delete(names, 0)

muX_train, muX_test, muy_train, muy_test = train_test_split(
        mush_data, mush_tar, test_size = .3)

grade = pd.read_csv('student-mat.csv', sep=";", header=0)
grade = pd.get_dummies(grade)
grade = scale(grade)
gradearr = np.array(grade)
grade_data, grade_tar = gradearr[:, :-1], gradearr[:, -1]

gxtr, gxte, gytr, gyte = train_test_split(
            grade_data, grade_tar, test_size = (0.3))

#commented out because it adds a lot of time to the run
"""
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
mx_train, mx_test = X[:60000], X[60000:]
my_train, my_test = y[:60000], y[60000:]
"""

# mlp = MLPClassifier(hidden_layer_sizes=(500,200,50), max_iter=150, alpha=1e-4,
#                        solver='sgd', verbose=10, tol=1e-5, random_state=1,
#                        learning_rate_init=.01, activation='tanh')

def classify(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=10, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.01, activation='tanh')
    mlp.fit(X_train, y_train)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

def main():
    #classify(mx_train, mx_test, my_train, my_test)
    #classify(muX_train, muX_test, muy_train, muy_test)
    #classify(gxtr, gxte, gytr,gyte)
    classify(chX_train, chX_test, chy_train, chy_test)
    
if __name__ == "__main__":
    main()
