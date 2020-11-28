import main.recorder as recorder
import main.preprocessUnit as preprocess
import main.featureExtractor as extractor
import main.command as command
import gui.gui as gui


#  manages all object used in this program  
class Manager:
  def __init__(self, FeatureExtractor, CommandManager, GUISmartHome):
    self.recorder = recorder.Recorder()
    self.preprocessUnit = preprocess.PreprocessUnit()
    self.featureExtractor = extractor.FeatureExtractor
    self.commandManager = CommandManager
    self.GUI = GUISmartHome
  
  # Runs gui commands if any are acquired in self.GUIQueue
  # check windows gui events
  def guiThread(self):
    while True:
      if self.GUI.isCommandAvailable():
        self.GUI.handle()
      self.GUI.checkEvents()

  def recordingThread(self):
    while True:
      if self.recorder.isAudioLevelAboveThreshold():
        self.recorder.runAcquisition()

  # Recognize recording and exchanges information between all objects  
  def dataCalculationThread(self):
    while True:
      if self.recorder.isDataAvailable():
        data = self.recorder.exportRecording()
        data = self.preprocessUnit.process(data)
        data = self.featureExtractor.extract(data)
        command = self.CommandManager.recognize(data)
        self.GUI.putIntoQueue(command)

  def trainThread(self):
    pass
    # Discuss with the Team how should we train data
        