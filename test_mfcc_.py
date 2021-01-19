from python_speech_features import mfcc
import numpy as np
from matplotlib import pyplot as plt
from main.featureExtractor import MFCC
from main.preprocessUnit import PreprocessUnit
import soundfile as sf
import logging
import os

# Clearing the logs
open('logging.log', 'w').close()
logging.basicConfig(filename = 'logging.log', level = logging.DEBUG)
logging.info(" Starting MFCC TEST")

audioData, fs = sf.read(os.getcwd() + r"\database\Aleksandra Rogowiec\1\naprzod.wav")

segmentTime = 0.030

preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.055497, samplingFrequency=fs, segmentTime=segmentTime)
audioDataSegments = preprocessUnit.process(audioData)

segmentOverlap = segmentTime / 2

audioData = np.array(preprocessUnit.downsample(audioData))
print(audioData.shape)
mfccFeaturesRef = mfcc(audioData, samplerate=8000, winlen=segmentTime, winstep=segmentOverlap,
                       nfilt=26, nfft=1024, numcep=13, preemph=0.95, ceplifter=1, appendEnergy=True)
mfccFeaturesRef = mfccFeaturesRef.flatten('C')


# samplesPerSegment = int(segmentTime*fs)
# samplesOverlap = int(segmentOverlap*fs)
# audioDataLen = len(audioData)
# numberOfSegments = int(audioDataLen/samplesOverlap)-1
# audioDataSegments = np.zeros((samplesPerSegment, numberOfSegments))

# for i in range(numberOfSegments):
#     audioDataSegments[:, i] += (audioData[i * samplesOverlap: i * samplesOverlap + samplesPerSegment])

obj = MFCC(audioDataSegments, samplerate=8e3, numberOfCepstras=13, numberOfMelFilters=26,
           numberOfFrequencyBins=1024)
mfccFeaturesTest = obj.extract()
mfccFeaturesTest = mfccFeaturesTest.flatten('F')

plt.plot(mfccFeaturesRef, c='blue', label='ref współczynniki mel-cepstralne')
plt.plot(mfccFeaturesTest, c='red', label='obliczone współczynniki mel-cepstralne')
plt.legend()
# plt.show()

