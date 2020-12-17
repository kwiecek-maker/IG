"""
comparision mfcc function from python_speech_features library with own implementation

"""

from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
from matplotlib import pyplot as plt
from main.featureExtractor import MFCC

fs, audioData = wavfile.read(r"test\wyłącz.wav")

segmentTime = 0.02
segmentOverlap = 0.01

mfccFeaturesRef = mfcc(audioData, samplerate=fs, winlen=segmentTime, winstep=segmentOverlap,
                       nfilt=26, nfft=1024, numcep=13, preemph=0, ceplifter=0, appendEnergy=True)
mfccFeaturesRef = mfccFeaturesRef.flatten('C')

samplesPerSegment = int(segmentTime*fs)
samplesOverlap = int(segmentOverlap*fs)
audioDataLen = len(audioData)
numberOfSegments = int(audioDataLen/samplesOverlap)-1

audioDataSegments = np.zeros((samplesPerSegment, numberOfSegments))
for i in range(numberOfSegments):
    audioDataSegments[:, i] += (audioData[i * samplesOverlap: i * samplesOverlap + samplesPerSegment])

obj = MFCC(audioDataSegments, samplerate=fs, numberOfCepstras=13, numberOfMelFilters=26,
           numberOfFrequencyBins=1024)
mfccFeaturesTest = obj.exctract()
mfccFeaturesTest = mfccFeaturesTest.flatten('F')

plt.plot(mfccFeaturesRef, label='referencyjne współczynniki mel-cepstralne')
plt.plot(mfccFeaturesTest, c='red', label='obliczone współczynniki mel-cepstralne')
plt.legend()
plt.show()
