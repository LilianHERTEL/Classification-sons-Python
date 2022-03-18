import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

#npArray = np.array(fftabs[:int(freqs.size/2)])
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


samplerate, data = wavfile.read("bass_acoustic_000-024-127.wav")
samples = data.shape[0]
from scipy.fftpack import fft,fftfreq
datafft = fft(data)
#Get the absolute value of real and complex component:
fftabs = abs(datafft)
freqs = fftfreq(samples,1/samplerate)
plt.plot(freqs,fftabs)
plt.xlim( [10, samplerate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])

var1=nMaxInArray(np.array(fftabs[:int(freqs.size/2)]),10)


    

