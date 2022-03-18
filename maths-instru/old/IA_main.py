import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import os
import json
from sklearn.model_selection import train_test_split

chemin = "C:/Users/Hector/.spyder-py3/"
N = 10
JSONdata = {}
count = 0

#with open(chemin+'dataFULL_BASS.json') as json_file:
#    JSONdata['soundsBass'] = json.load(json_file)['soundsBass']
#    print(pd.Series(JSONdata['soundsBass']))
print("OUVERTURE FILE")
with open(chemin+'data_FULL_Version_2.json') as json_file:
    JSONdata['sounds'] = json.load(json_file)['sounds']
    #print(pd.Series(JSONdata['soundsFlute']))
print("FIN OUVERTURE FILE")
Y = np.array([])
X = np.array(np.empty(shape=[0, N]))
print(X.shape)
print("DEBUT AJOUT DANS ARRAY")
for arr in JSONdata['sounds']:
    a = np.array(arr['maxs'])
    X = np.concatenate((X,[a]))
    Y = np.append(Y,arr['label'])
    count += 1
    if count%100 == 0:
        print(count)
        
print("FIN AJOUT DANS ARRAY")
    
#for arr in JSONdata['soundsFlute']:
#    a = np.array(arr['maxs'])
#    X = np.concatenate((X,[a]))
#    Y = np.append(Y,arr['label'])


Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size= 0.3, random_state=0)
print(Xtrain.shape)
print(Xtest.shape)
print('pourcentage:' ,Xtrain.shape[0]/X.shape[0])
print("DEBUT TRAIN")
#DECISION TREE

from sklearn.tree import DecisionTreeClassifier
Arbre_decision = DecisionTreeClassifier(random_state=0, max_depth=20)
clf = Arbre_decision.fit(Xtrain, ytrain)
print("FIN TRAIN")
from sklearn.metrics import accuracy_score
ypredit = clf.predict(Xtest)
print(accuracy_score(ytest, ypredit))

from sklearn import metrics
print(metrics.confusion_matrix(ytest, ypredit))

#KNEIGHBORS

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
clf = KNN.fit(Xtrain, ytrain)
ypredit = clf.predict(Xtest)
print("KNEIGHBORS : ")
print(accuracy_score(ytest, ypredit))



'''
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,copy_X_train=False,random_state=0)
clf = gpc.fit(Xtrain, ytrain)
gpc.score(Xtrain, ytrain)
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(Xtrain, ytrain)
RandomForestClassifier(max_depth=10, random_state=0)
print(clf.feature_importances_)
ypredit = clf.predict(Xtest)
print(accuracy_score(ytest, ypredit))



'''


















