from main.preprocessUnit import PreprocessUnit
from main.classificator import GMM
from main.command import CommandReadingFactorGMM
from main.command import Command
from main.command import CommandManager
from main.featureExtractor import MFCC

import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np

trainingDataPath = os.getcwd() + r"\trainningData\commands52Cepstras_112MelFIlters_1024FFT_3components_1000_iterations.txt"

classificator = GMM(n_components=3, max_iterations=1000)
commandFactory = CommandReadingFactorGMM(classificator, trainingDataPath)
commandFactory.readCommands()
commandManager = CommandManager(saveTrainedDataToCSV=False)
preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.8, downsamplingFrequency=8e3)
commandManager.acquireCommands(commandFactory.getCommandList(preprocessUnit))


audio1, samplerate1 = sf.read(os.getcwd() + r"\database\Aleksandra Rogowiec\1\naprzod.wav")
audio1PreProcessed = preprocessUnit.process(audio1)
mfcc1 = MFCC(audio1PreProcessed, numberOfCepstras=56, numberOfMelFilters=112,numberOfFrequencyBins=8192)
exportedFeatures1 = mfcc1.extract()
output1 = commandManager.recognize(exportedFeatures1)

researchedClassificator = commandManager.commands[0].classificator
y = []
t = np.arange(min(exportedFeatures1.flatten('F')), max(exportedFeatures1.flatten('F')), 1e-3)
for time in t:
   y.append(researchedClassificator.getProbabilityAtValue(time))
y = np.array(y)


# values, binsSides = np.histogram(exportedFeatures1.flatten('F'), density=True, bins=100)
# maxProbability = np.max(values)
# maxPSD = np.max(y)

y = y - np.mean(y)

plt.figure(figsize=(10, 10))
plt.hist(exportedFeatures1.flatten('F'), density=True, bins=300, color='b')
plt.plot(t, y, 'r--')

researchedClassificator = commandManager.commands[5].classificator
y = []
t = np.arange(min(exportedFeatures1.flatten('F')), max(exportedFeatures1.flatten('F')), 1e-3)
for time in t:
   y.append(researchedClassificator.getProbabilityAtValue(time))
y = np.array(y)

y = y - np.mean(y)
plt.hist(exportedFeatures1.flatten('F'), density=True, bins=100)
plt.plot(t, y, 'g--')
plt.legend(["model \"naprzod\"", "model start \"start\"", "histogram \"naprzod\""])

plt.show()

assert "naprzod" == output1


# audio2, samplerate2 = sf.read(os.getcwd() + r"\database\Aleksandra Rogowiec\1\odbierz.wav")
# audio2PreProcessed = preprocessUnit.process(audio2)
# output2 = commandManager.recognize(exportedFeatures1)
# mfcc2 = MFCC(audio2PreProcessed)
# exportedFeatures2 = mfcc2.extract()


# assert "odbierz" == output2












