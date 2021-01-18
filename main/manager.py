import main.recorder as recorder
import main.preprocessUnit as preprocess
import main.featureExtractor as extractor
import main.command as command
from main.classificator import FakeClassificator
import logging

# Manages all object used in Program
class Manager:
  def __init__(self, CommandManager, GUISmartHome):
    # self.recorder = recorder.Recorder(thresholdLevel=0.2)
    self.recorder = recorder.FakeRecorder('database', recordingAcquisitionFrequency=0.5)
    self.preprocessUnit = preprocess.PreprocessUnit(desiredLoudnessLevel=1.0, downsamplingFrequency=8e3)
    self.commandManager = CommandManager
    self.GUI = GUISmartHome

  def acquiringDataThread(self):
    commandFactory = command.CommandFactory('databases', FakeClassificator)
    commandFactory.readCommands()
    self.commandManager.acquireCommands(commandFactory.getCommandList)

  # Runs gui commands if any are acquired in self.GUIQueue
  # check windows gui events
  def guiLoop(self):
    if self.GUI.isCommandAvailable():
      self.GUI.handle()
    self.GUI.checkEvents()

  def recordingThread(self):
    self.recorder.run()

  # Recognize recording and exchanges information between all objects
  def dataCalculationLoop(self):
    if self.recorder.isDataAvailable():

      data = self.recorder.exportRecording()
      data = self.preprocessUnit.process(data)
      data = extractor.MFCC(data)
      command = self.commandManager.recognize(data)
      self.GUI.putIntoQueue(command)

  def trainThread(self):
    for _ in range(10):
      logging.info("training...")

#EOF
