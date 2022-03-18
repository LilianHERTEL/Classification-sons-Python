import numpy as np
#import pandas as pd
#import sklearn as sk
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
import os
import json
from scipy.fftpack import fft,fftfreq

JSONDatas = {}
JSONDatas['sounds'] = []
N = 10
count = 0

def createJson(label,arrayMax) :
    JSONDatas['sounds'].append({
        'label' : label,   
        'maxs' : arrayMax
        })
    
def saveJson(data) : 
    with open('data_Full_Version_2.json','w') as outFile:
        json.dump(data,outFile)
  
    

def nMaxInArray(array, n):
    npArray = array
    maxisRetenus = []
    SomMaxisRetenus = []
    NombrMaxisR = []
    Niveau = 0.3e+08
    isgood = False
    while(len(maxisRetenus) < n):
        for i in range(npArray.size):
            if npArray[i] > Niveau :
                #print("= " + str(npArray[i]) + " en : " + str(i))
                #print("    => "+ str(freqs[i]))
                isgood = False
                for j in range(len(maxisRetenus)):
                    if maxisRetenus[j]*1.1 >  freqs[i] and maxisRetenus[j]*0.9 < freqs[i]:
                        SomMaxisRetenus[j] += freqs[i]
                        NombrMaxisR[j] += 1
                        maxisRetenus[j] = SomMaxisRetenus[j]/NombrMaxisR[j]
                        isgood = True
                        break
                if isgood == False: 
                    SomMaxisRetenus.append(freqs[i])
                    maxisRetenus.append(freqs[i])
                    NombrMaxisR.append(1)
    
        Niveau = Niveau*0.75
    maxisRetenus.sort()
    res = maxisRetenus[:n]
    return res


str1 = "D:/Cours/2A/ML/nsynth-train/audio/"
str2 = "D:/Cours/2A/ML/PythonProj1/PythonApplication1/PythonApplication1/sons/"
str3 = "D:/Cours/2A/ML/nsynth-train/guitarSample/"
print('DEBUT LISTDIR')
entries = os.listdir("C:/Users/Lilian/Documents/IUT/Cours/2A/Modelisations_Mathematiques/Projet_Sons/nsynth-train.jsonwav/nsynth-train/audio")
print('FIN LISTDIR')
nombreEntries = entries.count
print('DEBUT FOR : ' + str(nombreEntries) + ' entries')
for entry in entries:
    count += 1
    if(count%50 == 0):
        print(count)
    #if entry.split('_')[0] != "organ":
    #    continue
    
    #print(entry)
    samplerate, data = wavfile.read(str1 +entry)
    
    
    
    samples = data.shape[0]
    datafft = fft(data)
    #Get the absolute value of real and complex component:
    fftabs = abs(datafft)
    freqs = fftfreq(samples,1/samplerate)
    #plt.plot(freqs,fftabs)
    #plt.xlim( [10, samplerate/2] )
    #plt.xscale( 'log' )
    #plt.grid( True )
    #plt.xlabel( 'Frequency (Hz)' )
   #plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
    
    labelFile = entry.split('_')[0] 
    createJson(labelFile,nMaxInArray(np.array(fftabs[:int(freqs.size/2)]),N))
    
    #print(JSONDatas)
    
saveJson(JSONDatas)
print('FIN FOR')
# ON A TOUTES LES BASSES JUSQUA bass_electronic_015-023-050.wav



