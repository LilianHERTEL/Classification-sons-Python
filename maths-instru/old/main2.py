import matplotlib.pyplot as plot
from scipy.io import wavfile


samplingFrequency, signalData = wavfile.read('bass_acoustic_000-024-127.wav')

#plot.subplot(211)

#plot.title('Spectrogram of a wav file with piano music')


#plot.plot(signalData)

#plot.xlabel('Sample')

#plot.ylabel('Amplitude')

 

#plot.subplot(212)

var1 = plot.specgram(signalData,Fs=samplingFrequency)

#plot.xlabel('Time')

#plot.ylabel('Frequency')

#plot.show()

import matplotlib.pyplot as plt
import numpy as np
 

#X = np.clip(var1[0]*20,0,255)
X = var1[0]
mini = var1[0].min() 
maxi = var1[0].max() 

#plt.imshow(X)#cmap="gray"
plt.show()