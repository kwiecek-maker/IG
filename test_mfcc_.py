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

audioData, fs = sf.read(r"database\Aleksandra Rogowiec\1\naprzod.wav")
audioData2, fs = sf.read(r"C:\python\IG\database\Aleksandra_Pietrek\1\naprzod.wav")

segmentTime = 0.030

preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.055497, samplingFrequency=fs, segmentTime=segmentTime)
audioDataSegments = preprocessUnit.process(audioData)
audioDataSegments2 = preprocessUnit.process(audioData2)

segmentOverlap = segmentTime / 2

audioData = np.array(preprocessUnit.downsample(audioData))
print(audioData.shape)
mfccFeaturesRef = mfcc(audioData, samplerate=8000, winlen=segmentTime, winstep=segmentOverlap,
                       nfilt=26, nfft=1024, numcep=12, preemph=0.95, ceplifter=1, appendEnergy=True)

print("Mccf Features referene shape: %d, %d" % (mfccFeaturesRef.shape[0], mfccFeaturesRef.shape[1]))
mfccFeaturesRef = mfccFeaturesRef.flatten('C')


# samplesPerSegment = int(segmentTime*fs)
# samplesOverlap = int(segmentOverlap*fs)
# audioDataLen = len(audioData)
# numberOfSegments = int(audioDataLen/samplesOverlap)-1
# audioDataSegments = np.zeros((samplesPerSegment, numberOfSegments))

# for i in range(numberOfSegments):
#     audioDataSegments[:, i] += (audioData[i * samplesOverlap: i * samplesOverlap + samplesPerSegment])

obj = MFCC(audioDataSegments, samplerate=8e3, numberOfCepstras=12, numberOfMelFilters=20,
           numberOfFrequencyBins=1024)
mfccFeaturesTest1 = obj.extract()
print("Mccf Features test shape: %d, %d" % (mfccFeaturesTest1.T.shape[0], mfccFeaturesTest1.T.shape[1]))

obj = MFCC(audioDataSegments2, samplerate=8e3, numberOfCepstras=12, numberOfMelFilters=20,
           numberOfFrequencyBins=1024)
mfccFeaturesTest2 = obj.extract()
print("Mccf Features test shape2: %d, %d" % (mfccFeaturesTest2.T.shape[0], mfccFeaturesTest2.T.shape[1]))

mfccFeaturesTest3 = np.concatenate((mfccFeaturesTest1, mfccFeaturesTest2), axis=1)

print("Mccf Features test shape3: %d, %d" % (mfccFeaturesTest3.T.shape[0], mfccFeaturesTest3.T.shape[1]))


# mfccFeaturesTest = mfccFeaturesTest.flatten('F')

# plt.plot(mfccFeaturesRef, c='blue', label='ref współczynniki mel-cepstralne')
# plt.plot(mfccFeaturesTest, c='red', label='obliczone współczynniki mel-cepstralne')
print(mfccFeaturesTest2[-3].flatten('F')[-60:])
plt.show()

assert False == True

