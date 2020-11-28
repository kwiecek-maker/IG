import main.featureExtractor as extractor
import main.command as command
import main.manager as manager
import GUI.gui as GUI

import threading 
import logging
import time
import keyboard

# Clearing the logs
open('logging.log', 'w').close()

FeatureExtractor = extractor.MFCC()
CommandManager = command.CommandManager()
Gui = GUI.GUISmartHome()

Manager = manager.Manager(FeatureExtractor, CommandManager, Gui) 
  
logging.basicConfig(filename = 'logging.log', encoding = 'utf-8', level = logging.DEBUG)
logging.info("Starting program")

def trainThread():
  t = threading.current_thread()
  logging.info("Training %s started", t.getName())
  Manager.trainThread()
  logging.info("Training %s finished", t.getName())

def guiThread():
  t = threading.current_thread()
  logging.info("Gui %s started", t.getName())
  while True:
    Manager.guiLoop()
    if (keyboard.is_pressed('q')):
      break
  logging.info("Gui %s finished", t.getName())

def dataCalculationThread():
  t = threading.current_thread()
  logging.info("Data Calculation %s started", t.getName())
  while True:
    Manager.dataCalculationLoop()
    if (keyboard.is_pressed('q')):
      break
  logging.info("Gui %s finished", t.getName())

def recordingThread():
  t = threading.current_thread()
  logging.info("Recording %s started", t.getName())
  while True:
    Manager.recordingLoop()
    if (keyboard.is_pressed('q')):
      break
  logging.info("Recording %s finished", t.getName())

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
