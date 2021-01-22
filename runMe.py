import main.featureExtractor as extractor
import main.command as command
import main.manager as manager
import main.classificator as classificators
import GUI.gui as GUI

import threading
import logging
import keyboard

# Clearing the logs
open('logging.log', 'w').close()

CommandManager = command.CommandManager()
Gui = GUI.GUISmartHome()
classificator = classificators.GMM(n_components=6, max_iterations=200)

Manager = manager.Manager(classificator ,CommandManager, Gui)
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
