from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
clf = KNN.fit(Xtrain10, ytrain)
ypredit = clf.predict(Xtest10)
print(accuracy_score(ytest, ypredit))
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
clf = KNN.fit(Xtrain, ytrain)
ypredit = clf.predict(Xtest)
print(accuracy_score(ytest, ypredit))
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,copy_X_train=False,random_state=0)
clf = gpc.fit(Xtrain10, ytrain)
clf = gpc.fit(Xtest, ytest)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(Xtrain, ytrain)
RandomForestClassifier(max_depth=2, random_state=0)
print(clf.feature_importances_)
print(clf.predict(Xtest))
print(accuracy_score(ytest, ypredit))
ypredit = clf.predict(Xtest)
print(accuracy_score(ytest, ypredit))

## ---(Fri Dec 27 18:07:34 2019)---
runfile('C:/Users/Hector/.spyder-py3/main2.py', wdir='C:/Users/Hector/.spyder-py3')
te = 1.0/rate
t = np.zeros(n)
for k in range(n):
    t[k] = te*k
    
figure(figsize=(12,4))
plot(t,data)
xlabel("t (s)")
ylabel("amplitude") 
axis([0,0.1,data.min(),data.max()])
grid()
def tracerSpectre(data,rate,debut,duree):
    start = int(debut*rate)
    stop = int((debut+duree)*rate)
    spectre = np.absolute(fft(data[start:stop]))
    spectre = spectre/spectre.max()
    n = spectre.size
    freq = np.zeros(n)
    for k in range(n):
        freq[k] = 1.0/n*rate*k
    vlines(freq,[0],spectre,'r')
    xlabel('f (Hz)')
    ylabel('A')
    axis([0,0.5*rate,0,1])
    grid()
    
figure(figsize=(12,4))
tracerSpectre(data,rate,0.0,0.5)
axis([0,5000,0,1])
figure(figsize=(12,4))
tracerSpectre(data,rate,2.0,0.5)
axis([0,5000,0,1])
tracerSpectre(data,rate,0.0,duree)
runfile('C:/Users/Hector/.spyder-py3/main2.py', wdir='C:/Users/Hector/.spyder-py3')
runfile('C:/Users/Hector/.spyder-py3/main3.py', wdir='C:/Users/Hector/.spyder-py3')
npArray.size
runfile('C:/Users/Hector/.spyder-py3/main3.py', wdir='C:/Users/Hector/.spyder-py3')
runfile('C:/Users/Hector/.spyder-py3/temp.py', wdir='C:/Users/Hector/.spyder-py3')

## ---(Sat Dec 28 00:12:49 2019)---
runfile('C:/Users/Hector/.spyder-py3/temp.py', wdir='C:/Users/Hector/.spyder-py3')

## ---(Sat Dec 28 00:58:23 2019)---
import os
entries = os.listdir("D:/Cours/2A/ML/nsynth-train/audio/")
runfile('C:/Users/Hector/.spyder-py3/temp.py', wdir='C:/Users/Hector/.spyder-py3')
runfile('C:/Users/Hector/.spyder-py3/IA_main.py', wdir='C:/Users/Hector/.spyder-py3')
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(Xtrain, ytrain)
RandomForestClassifier(max_depth=2, random_state=0)
print(clf.feature_importances_)
ypredit = clf.predict(Xtest)
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel,copy_X_train=False,random_state=0)
clf = gpc.fit(Xtrain, ytrain)
gpc.score(Xtrain, ytrain)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(Xtrain, ytrain)
RandomForestClassifier(max_depth=2, random_state=0)
print(clf.feature_importances_)
ypredit = clf.predict(Xtest)
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(Xtrain, ytrain)
RandomForestClassifier(max_depth=10, random_state=0)
print(clf.feature_importances_)
ypredit = clf.predict(Xtest)
print(accuracy_score(ytest, ypredit))