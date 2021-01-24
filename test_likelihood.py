from main.preprocessUnit import PreprocessUnit
from main.classificator import GMM
from main.command import CommandReadingFactorGMM
from main.command import Command
from main.command import CommandManager
from main.command import CommandFactory
from main.featureExtractor import MFCC
from main.manager import Manager
from main.recorder import FakeRecorder
import GUI.gui as GUI

import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np

# trainingDataPath = os.getcwd() + r"\trainningData\commands_12_Cepstras_24_MelFIlters_512_FFT_4_components_100_iterations.txt"

# classificator = GMM(n_components=4, max_iterations=1000)
# commandFactory = CommandReadingFactorGMM(classificator, trainingDataPath)
# commandFactory.readCommands()
# commandManager = CommandManager(saveTrainedDataToCSV=False)
# preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.8, downsamplingFrequency=8e3)
# commandManager.acquireCommands(commandFactory.getCommandList(preprocessUnit))

# plt.figure(figsize=(10, 10))

# t = np.arange(-30, 30, 1e-3)
# legend = []
# for command in commandManager.commands:
#    y = []
#    for time in t:
#       y.append(command.classificator.getProbabilityAtValue(time))
#    y = np.array(y) / max(y)
#    plt.plot(t, y, '--')
#    legend.append(command.name)

# plt.legend(legend)
# plt.show()

# Gui = GUI.GUISmartHome()
# classificator = GMM(n_components=4, max_iterations=100)
# trainedDataPath = os.getcwd() + r"\trainningData\commands_12_Cepstras_24_MelFIlters_512_FFT_4_components_100_iterations.txt"
# commandManager = CommandManager(saveTrainedDataToCSV=False, trainingDataDestinationPath=trainedDataPath)
# CommandFactory = CommandFactory('database', classificator) #! getting train data

# preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.8, downsamplingFrequency=12e3)
# recorder = FakeRecorder('database', 0.5)
# manager = Manager(classificator, commandManager, CommandFactory, preprocessUnit, Gui, recorder)

# manager.acquiringDataThread()
# mfccList = commandManager.commands[5].dataList

# print(mfccList[0].shape)
# output = np.concatenate((mfccList[0], mfccList[1]), axis=1)
# print(output.shape)

assert True == True


# audio2, samplerate2 = sf.read(os.getcwd() + r"\database\Aleksandra Rogowiec\1\odbierz.wav")
# audio2PreProcessed = preprocessUnit.process(audio2)
# output2 = commandManager.recognize(exportedFeatures1)
# mfcc2 = MFCC(audio2PreProcessed)
# exportedFeatures2 = mfcc2.extract()

# assert "odbierz" == output2