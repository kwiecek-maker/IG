from main.preprocessUnit import PreprocessUnit
from main.classificator import GMM
from main.command import CommandReadingFactorGMM
from main.command import Command
from main.command import CommandManager
from main.featureExtractor import MFCC

import soundfile as sf
import os

trainingDataPath = os.getcwd() + r"\trainningData\smartHomeCommands.txt"

classificator = GMM(n_components=12, max_iterations=2000)
commandFactory = CommandReadingFactorGMM(classificator, trainingDataPath)
commandFactory.readCommands()
commandManager = CommandManager(saveTrainedDataToCSV=False)
preprocessUnit = PreprocessUnit(desiredLoudnessLevel=0.8, downsamplingFrequency=8e3)
commandManager.acquireCommands(commandFactory.getCommandList(preprocessUnit))


audio1, samplerate1 = sf.read(os.getcwd() + r"\database\Aleksandra Rogowiec\1\naprzod.wav")
audio1PreProcessed = preprocessUnit.process(audio1)
mfcc1 = MFCC(audio1PreProcessed)
exportedFeatures1 = mfcc1.extract()
output1 = commandManager.recognize(exportedFeatures1)
assert "naprzod" == output1


audio2, samplerate2 = sf.read(os.getcwd() + r"database\Aleksandra Rogowiec\1\odbierz.wav")
audio2PreProcessed = preprocessUnit.process(audio2)
output2 = commandManager.recognize(exportedFeatures1)
mfcc2 = MFCC(audio2PreProcessed)
exportedFeatures2 = mfcc2.extract()
assert "odbierz" == output2












