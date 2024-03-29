import main.recorder as recorder
import main.preprocessUnit as preprocess
import main.featureExtractor as extractor
import main.command as command
from main.classificator import FakeClassificator
import logging
import time

# Manages all object used in Program
class Manager:
  def __init__(self, classificator, CommandManager, CommandFactory, preprocessUnit, GUISmartHome, recorder):
    # self.recorder = recorder.Recorder(thresholdLevel=0.2)
    self.recorder = recorder
    self.preprocessUnit = preprocessUnit
    self.commandManager = CommandManager
    self.commandFactory = CommandFactory
    self.classificator = classificator
    self.GUI = GUISmartHome

  def acquiringDataThread(self):
    self.commandFactory.readCommands()
    self.commandManager.acquireCommands(self.commandFactory.getCommandList(self.preprocessUnit))

  # Runs gui commands if any are acquired in self.GUIQueue
  # check windows gui events
  def guiLoop(self):
    if self.GUI.isCommandAvailable():
      self.GUI.handle()
    self.GUI.checkEvents()
    time.sleep(0.3)

  def recordingThread(self):
    self.recorder.run()

  # Recognize recording and exchanges information between all objects
  def dataCalculationLoop(self):
    if self.recorder.isDataAvailable():

      data, samplerate = self.recorder.exportRecording()
      data = self.preprocessUnit.process(data)
      data = extractor.ReferenceMFCC(data, samplerate)
      # data = extractor.MFCC(data,  samplerate=self.preprocessUnit.downsamplingFrequency, numberOfCepstras=12, numberOfMelFilters=24, numberOfFrequencyBins=512  )
      command = self.commandManager.recognize(data.extract().T)
      self.GUI.putIntoQueue(command)

  def trainThread(self):
    self.commandManager.train()

#EOF
