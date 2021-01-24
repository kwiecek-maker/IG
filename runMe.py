import main.featureExtractor as extractor
import main.command as command
import main.manager as manager
import main.classificator as classificators
import main.preprocessUnit as preprocess
import main.recorder as rec
import GUI.gui as GUI

import threading
import logging
import keyboard
import os

global DEBUG
DEBUG = True

# Clearing the logs
open('logging.log', 'w').close()

Gui = GUI.GUISmartHome()
# classificator = classificators.GMM(n_components=4, max_iterations=100)
classificator = classificators.ReferenceGMM(n_components=7, max_iter=300)
trainedDataPath = os.getcwd() + r"\trainningData\commands_12_Cepstras_24_MelFIlters_512_FFT_4_components_100_iterations.txt"

training = True

CommandManager = command.CommandManager(saveTrainedDataToCSV=False, trainingDataDestinationPath=trainedDataPath)
CommandFactory = None
if training:
  CommandFactory = command.CommandFactory('database', classificator) #! getting train data
else:
  CommandFactory = command.CommandReadingFactorGMM(classificator, trainedDataPath)

preprocessUnit = preprocess.PreprocessUnit(desiredLoudnessLevel=0.8, downsamplingFrequency=12e3)

recorder = rec.FakeRecorder('database', recordingAcquisitionFrequency=0.05)
Manager = manager.Manager(classificator ,CommandManager, CommandFactory, preprocessUnit, Gui, recorder)
logging.basicConfig(filename = 'logging.log', level = logging.DEBUG)
logging.info(" Starting program")

def acquiringDataThread():
  t = threading.current_thread()
  logging.info(" Acquiring Data %s has started", t.getName())
  Manager.acquiringDataThread()
  logging.info(" Acquiring Data %s has ended!", t.getName())

def trainThread():
  t = threading.current_thread()
  logging.info(" Training %s started", t.getName())
  Manager.trainThread()
  logging.info(" Training %s finished", t.getName())

def guiThread():
  t = threading.current_thread()
  logging.info(" Gui %s started", t.getName())
  while True:
    Manager.guiLoop()
    if (keyboard.is_pressed('q')):
      break
  logging.info(" Gui %s finished", t.getName())

def dataCalculationThread():
  t = threading.current_thread()
  logging.info(" Data Calculation %s started", t.getName())
  while True:
    Manager.dataCalculationLoop()
    if (keyboard.is_pressed('q')):
      break
  logging.info(" Gui %s finished", t.getName())

def recordingThread():
  t = threading.current_thread()
  logging.info(" Recording %s started", t.getName())
  Manager.recordingThread()
  logging.info(" Recording %s finished", t.getName())

def run():
  acquiringThread = threading.Thread(target=acquiringDataThread)
  acquiringThread.start()
  acquiringThread.join()

  TrainingThread = threading.Thread(target=trainThread)
  TrainingThread.start()
  TrainingThread.join()

  threadList = []
  threadList.append(threading.Thread(target=guiThread))
  threadList.append(threading.Thread(target=dataCalculationThread))
  threadList.append(threading.Thread(target=recordingThread))

  for thread in threadList:
    thread.start()

  for thread in threadList:
    thread.join()

if __name__ == "__main__":
  run()
