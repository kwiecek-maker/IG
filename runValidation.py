import main.command as command
import main.manager as Manager
import main.classificator as classificators
import main.preprocessUnit as preprocess
import main.recorder as Recorder
import GUI.gui as GUI

from main.featureExtractor import MFCC
from main.featureExtractor import ReferenceMFCC


import threading
import logging
import keyboard
import os

# Clearing the logs
open('logging.log', 'w').close()

Gui = GUI.GUISmartHome()
# classificator = classificators.GMM(n_components=4, max_iterations=100)
classificator = classificators.ReferenceGMM(n_components=5, max_iter=200)
trainedDataPath = os.getcwd() + r"\trainningData\commands_12_Cepstras_24_MelFIlters_512_FFT_4_components_100_iterations.txt"
CommandManager = command.CommandManager(saveTrainedDataToCSV=False, trainingDataDestinationPath=trainedDataPath)
# CommandFactory = command.CommandReadingFactorGMM(classificator, trainedDataPath)
CommandFactory = command.CommandFactory('database', classificator)
# preprocessUnit = preprocess.PreprocessUnit(desiredLoudnessLevel=0.8, downsamplingFrequency=12e3)
preprocessUnit = preprocess.FakePreprocessorUnit()

recorder = Recorder.ValidationRecorder('validationDatabase')
manager = Manager.Manager(classificator, CommandManager, CommandFactory, preprocessUnit, Gui, recorder)


logging.basicConfig(filename = 'logging.log', level = logging.DEBUG)
logging.info(" Starting Validation")
manager.acquiringDataThread()
manager.trainThread()
score = 0.0
numberOfTest = int(1e3)
for _ in range(numberOfTest):
   audio, samplerate, referenceName = recorder.exportRecording()
   preprocessedAudio = preprocessUnit.process(audio)
   # mfcc = MFCC(preprocessedAudio, numberOfCepstras=12, numberOfMelFilters=24, numberOfFrequencyBins=512)
   mfcc = ReferenceMFCC(preprocessedAudio, samplerate)
   outputName = CommandManager.recognize(mfcc.extract().T)
   logging.info("recording / recognised: \"%s\" / \"%s\"" % (referenceName, outputName))

   if outputName == referenceName:
      score += 1

message = "overall score: %f" %(score / numberOfTest)
logging.info(message)
print(message)
logging.info("ending validation")







