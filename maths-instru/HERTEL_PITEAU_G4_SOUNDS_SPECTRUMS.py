# -*- coding: utf-8 -*-
"""
@title: mathematical modeling project
@subject: sounds supervised classification
@date: Jan 2020
@authors: Hector PITEAU and Lilian HERTEL

@dataset: NSynth dataset
"""

import os
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn import preprocessing

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from datetime import datetime
import sys
import math

spectrumData = []
yData = []



'''************''''''************'''
''' PARAMETERS ''''''************'''
'''************''''''************'''

# The directory where the .wav files are located
directory = "C:/Users/Lilian/Documents/IUT/Cours/2A/Modelisations_Mathematiques/Projet_Sons/nsynth-train.jsonwav/nsynth-train/audio/"

# Instruments we want to work on (instruments that will be added in yData and spectrumData)
# 1 if we want to work on the instrument, 0 otherwise
useDict = dict()
useDict["bass"] = 1
useDict["brass"] = 1
useDict["flute"] = 1
useDict["guitar"] = 1
useDict["keyboard"]= 1
useDict["mallet"] = 1
useDict["organ"] = 1
useDict["reed"]  = 1
useDict["string"]  = 1
useDict["synth"]  = 1
useDict["vocal"] = 1

# Number of samples of each instrument we want to work on
# Value > 65474 to use all samples
countMax = 1000

# To normalize the spectrograms (values between 0 and 1)
# True or False
use_normalization = True

# To resize the shape of the spectrograms (divide by 3)
# True or False
use_resizing = True

# Classifier choice:
# Choosed classifier should be set to True
# The first one will be used if more than 1 classifier are set to True
use_decision_tree = False
use_random_forest = True

'''************''''''************'''
'''************''''''************'''
'''************''''''************'''



# Number of samples in the dataset, for each instrument
nbDict = dict()
nbDict["bass"] = 65474
nbDict["brass"] = 12675
nbDict["flute"] = 8773
nbDict["guitar"] = 32690
nbDict["keyboard"]= 51821
nbDict["mallet"] = 34201
nbDict["organ"] = 34477
nbDict["reed"]  = 13911
nbDict["string"]  = 19474
nbDict["synth"]  = 5501
nbDict["vocal"] = 10208

# Real number of samples that will be used for each instrument (because for some instruments we have less samples than countMax)
maxDict = dict()
maxDict['bass'] = min(countMax, nbDict["bass"]) * useDict["bass"]
maxDict['brass']  = min(countMax, nbDict["brass"]) * useDict["brass"]
maxDict['flute'] = min(countMax, nbDict["flute"]) * useDict["flute"]
maxDict['guitar']  = min(countMax, nbDict["guitar"]) * useDict["guitar"]
maxDict['keyboard']  = min(countMax, nbDict["keyboard"]) * useDict["keyboard"]
maxDict['mallet'] = min(countMax, nbDict["mallet"]) * useDict["mallet"]
maxDict['organ'] = min(countMax, nbDict["organ"]) * useDict["organ"]
maxDict['reed']  = min(countMax, nbDict["reed"]) * useDict["reed"]
maxDict['string']  = min(countMax, nbDict["string"]) * useDict["string"]
maxDict['synth'] = min(countMax, nbDict["synth"]) * useDict["synth"]
maxDict['vocal']  = min(countMax, nbDict["vocal"]) * useDict["vocal"]

# Counters: number of each instrument that has already been added to yData and spectrumData
countDict = dict()
countDict['bass'] = 0
countDict['brass'] = 0
countDict['flute'] = 0
countDict['guitar'] = 0
countDict['keyboard'] = 0
countDict['mallet'] = 0
countDict['organ'] = 0
countDict['reed'] = 0
countDict['string'] = 0
countDict['synth'] = 0
countDict['vocal'] = 0

nbSamplesDone = 0
nbSamples = maxDict['bass'] + maxDict['brass'] + maxDict['flute'] + maxDict['guitar'] + maxDict['keyboard'] + maxDict['mallet'] + maxDict['organ'] + maxDict['reed'] + maxDict['string'] + maxDict['synth'] + maxDict['vocal']
nbSamplesModulo = math.floor(nbSamples/1000)

# This function reshapes a 2D array: we use it to divide the shape of each spectrogram by 3
def resize(arr, new_shape):
    """Rebin 2D array arr to shape new_shape by averaging."""
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

# Prints the given time and message
def display_time(time, message):
    print("\n" + message)
    print("------------------------------")
    print(str(time) + "\n")

time_start = datetime.now()
display_time(time_start, "APPLICATION START")

entries = os.listdir(directory)
nombreEntries = entries.count

time_start_spectrograms = datetime.now()
display_time(time_start_spectrograms, "Spectrograms computing -- START")

sys.stdout.write("\rPROGRESSION: 0%")
sys.stdout.flush()
for entry in entries:    
    # The sound class, found by trimming the file name before the first "_" character
    sound_class = entry.split('_')[0]
    
    # Skips the sound if it shouldn't be used (as defined in PARAMETERS)
    if sound_class in countDict:
        if (not useDict[sound_class]):
            continue
        elif (countDict[sound_class] >= maxDict[sound_class]):
            continue
        else:
            countDict[sound_class] += 1
            yData.append(sound_class)

    nbSamplesDone += 1
    if nbSamplesModulo < 1:
        nbSamplesModulo = 1
    if(nbSamplesDone%(nbSamplesModulo) == 0):
        sys.stdout.write("\rPROGRESSION: %.1f%s" % ((nbSamplesDone/nbSamples)*100, "%"))
        sys.stdout.flush()        
    
    samplingFrequency, signalData = wavfile.read(directory + entry)
    f, t, Sxx = signal.spectrogram(signalData, samplingFrequency)
    if (use_normalization):
        Sxx = preprocessing.normalize(Sxx, norm='l2')
    if (use_resizing):
        Sxx = resize(Sxx, (43,95)) #43, 95 = shape/3
    spectrum.append(Sxx.flatten())
    
    # Breaks when all spectrograms have been computed
    if countDict['bass'] == maxDict['bass'] and countDict['brass'] == maxDict['brass'] and countDict['flute'] == maxDict['flute'] and countDict['guitar'] == maxDict['guitar'] and countDict['keyboard'] == maxDict['keyboard'] and countDict['mallet'] == maxDict['mallet'] and countDict['organ'] == maxDict['organ'] and countDict['reed'] == maxDict['reed'] and countDict['string'] == maxDict['string'] and countDict['synth'] == maxDict['synth'] and countDict['vocal'] == maxDict['vocal']:
        break

sys.stdout.write("\rPROGRESSION: 100.0%")
sys.stdout.flush()

print("\n")
time_end_spectrograms = datetime.now()
display_time(time_end_spectrograms, "Spectrograms computing -- END");

time_interval_spectrograms = time_end_spectrograms - time_start_spectrograms
display_time(time_interval_spectrograms, "Spectrograms computing -- TIME");

Xtrain, Xtest, ytrain, ytest = train_test_split(spectrum, yData, test_size= 0.2, random_state=0)

time_start_training = datetime.now()
display_time(time_start_training, "Training and testing -- START")

if use_decision_tree:
    print("Decision tree -- START")
    Arbre_decision = DecisionTreeClassifier(random_state=0, max_depth=20)
    clf = Arbre_decision.fit(Xtrain, ytrain)
    ypredit = clf.predict(Xtest)
    print("Decision tree -- END")
elif use_random_forest:
    print("Random forest -- START")
    clf = RandomForestClassifier(max_depth=20, random_state=0)
    clf.fit(Xtrain, ytrain)
    RandomForestClassifier(max_depth=20, random_state=0)
    print(clf.feature_importances_)
    ypredit = clf.predict(Xtest)
    print("Random forest -- END")
else:
    display_time(datetime.now(), "ERROR: Select a model")

time_end_training = datetime.now()
display_time(time_end_training, "Training and testing -- END")

time_interval_training = time_end_training - time_start_training
display_time(time_interval_training, "Training and testing -- TIME")

# Calculates and prints the total time
time_interval_total = time_end_training - time_start
display_time(time_interval_total, "Total -- TIME")

if use_decision_tree or use_random_forest:
    print("******************************")
    print("CLASSIFICATION ACCURACY:")
    print(accuracy_score(ytest, ypredit))
    print("******************************\n")
    
    print("Confusion matrix:")
    theLabels = []
    for key,val in useDict.items():
        if val == 1:
            theLabels.append(key)
            
    array = (metrics.confusion_matrix(ytest, ypredit,labels=theLabels))
    leDataFrame = pd.DataFrame(array, index = theLabels,columns = theLabels)
    plt.figure(figsize = (10,10))
    ax = sn.heatmap(leDataFrame, annot=True,cmap="Blues",square=True,fmt='d')
    ax.set_ylim(len(array), -0.5)