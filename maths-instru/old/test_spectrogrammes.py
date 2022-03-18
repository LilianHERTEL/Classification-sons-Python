# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:49:23 2020

@author: Lilian
"""

#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot

from scipy.io import wavfile

# Read the wav file (mono)
samplingFrequency, signalData = wavfile.read('bass_acoustic_000-024-127.wav')
# Plot the signal read from wav file
#plot.subplot(211)
#plot.title('Spectrogram of a wav file with piano music')
#plot.plot(signalData)
#plot.xlabel('Sample')
#plot.ylabel('Amplitude')
#plot.subplot(212)
print(samplingFrequency)
print(signalData)
res = plot.specgram(signalData,Fs=samplingFrequency)
spectrum = res[0]
res[0]

#plot.xlabel('Time')

#plot.ylabel('Frequency')

 

#plot.show()




