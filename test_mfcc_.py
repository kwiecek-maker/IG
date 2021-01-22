from python_speech_features import mfcc
import numpy as np
from matplotlib import pyplot as plt
from main.featureExtractor import MFCC
from main.preprocessUnit import PreprocessUnit
from main.classificator import GMM
import soundfile as sf
import logging
import os

# Clearing the logs
open('logging.log', 'w').close()

logging.basicConfig(filename = 'logging.log', level = logging.DEBUG)
logging.info(" Starting MFCC TEST")

audioData, fs = sf.read(r"database\Aleksandra Rogowiec\1\naprzod.wav")

preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.8, samplingFrequency=fs)
audioDataSegments = preprocessUnit.process(audioData)

obj = MFCC(audioDataSegments, samplerate=8e3, numberOfCepstras=12, numberOfMelFilters=20,
           numberOfFrequencyBins=1024)

data = obj.extract().flatten('F')

classificator = GMM(n_components=12, max_iterations=100)
classificator.train([data])
t = np.arange(min(data), max(data), 1e-3)
y = []
for time in t:
   y.append(classificator.getProbabilityAtValue(time))



plt.figure()
plt.hist(data, density=True, bins=100)
plt.plot(t, y, 'r--')
# plt.show()


assert True == True

