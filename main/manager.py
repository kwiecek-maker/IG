import main.recorder as recorder
import main.preprocessUnit as preprocess

import threading
import logging

#  manages all object used in this program  
class Manager:
  def __init__(self, FeatureExtractor, CommandManager, GUISmartHome):
    self.recorder = recorder.Recorder(thresholdLevel = 0.5)
    self.preprocessUnit = preprocess.PreprocessUnit(desiredLoudnessLevel=1.0, downsamplingFrequency=8e3)
    self.featureExtractor = FeatureExtractor
    self.commandManager = CommandManager
    self.GUI = GUISmartHome
  
  # Runs gui commands if any are acquired in self.GUIQueue
  # check windows gui events
  def guiLoop(self):
    if self.GUI.isCommandAvailable():
      self.GUI.handle()
    self.GUI.checkEvents()
    
  def recordingLoop(self):
    if self.recorder.isAudioLevelAboveThreshold():
      self.recorder.runAcquisition()

  # Recognize recording and exchanges information between all objects  
  def dataCalculationLoop(self):
    if self.recorder.isDataAvailable():
      data = self.recorder.exportRecording()
      data = self.preprocessUnit.process(data)
      data = self.featureExtractor.extract(data)
      command = self.CommandManager.recognize(data)
      self.GUI.putIntoQueue(command)

  def trainThread(self):
    for _ in range(10):
      logging.info("training...")
    # Discuss with the Team how should we train data
        