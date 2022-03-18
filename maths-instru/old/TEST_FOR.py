# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:30:22 2020

@author: Lilian
"""
import os
from scipy.io import wavfile
import matplotlib.pyplot as plot
import numpy as np


spectrum = np.array([])

countA = 0
countB = 0
directory="C:/Users/Lilian/Documents/IUT/Cours/2A/Modelisations_Mathematiques/Projet_Sons/nsynth-train.jsonwav/nsynth-train/audio/"
print('DEBUT LISTDIR')
entries = os.listdir(directory)
print('FIN LISTDIR')
nombreEntries = entries.count
print('DEBUT FOR : ' + str(nombreEntries) + ' entries')
for entry in entries:

    if (entry.split('')[0] == "mallet" and countA == 4000) :
        continue
    if (entry.split('')[0] == "organ" and countB == 4000) :
        continue

    if entry.split('')[0] != "mallet" and entry.split('')[0] != "organ":
        continue

    if entry.split('')[0] == "mallet" :
        countA += 1
    if entry.split('')[0] != "organ" :
        countB += 1

    if(countA%50 == 0):
        print(countA)

    samplingFrequency, signalData = wavfile.read(directory + entry)
    res = plot.specgram(signalData,Fs=samplingFrequency)

    spectrum = np.append(spectrum,res[0].flatten())
    if countA == 4000 and countB == 4000:
        break
print('FIN FOR')