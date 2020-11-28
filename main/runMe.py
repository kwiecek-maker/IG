import main.featureExtractor as extractor
import main.command as command
import main.manager as manager
import gui.gui as GUI

import threading 
import logging

FeatureExtractor = extractor.MFCC
CommandManager = command.CommandManager
Gui = GUI.GUISmartHome

Manager = manager.Manager(FeatureExtractor, CommandManager, Gui) 
  
logging.basicConfig(filename = 'logging.txt', encoding = 'utf-8', level = logging.DEBUG)
logging.info("Starting program")

def trainThread():
  t = threading.current_thread()
  logging.info("Training Thread - number %s started", t.getName())
  manager.trainThread()
  logging.info("Training Thread: %s finished", t.getName())

def guiThread():
  t = threading.current_thread()
  logging.info("Gui Thread - number %s started", t.getName())
  Manager.guiThread()
  logging.info("Gui Thread: %s finished", t.getName())

def dataCalculationThread():
  t = threading.current_thread()
  logging.info("Data Calculation Thread - number %s started", t.getName())
  manager.dataCalculationThread()
  logging.info("Gui Thread: %s finished", t.getName())

def recordingThread():
  t = threading.current_thread()
  logging.info("Recording Thread - number %s started", t.getName())
  manager.recordingThread()
  logging.info("Recording Thread: %s finished", t.getName())

def run():
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
  run(); 
