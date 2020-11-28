# Recorder is listening for commands, starts/stop aquairing them
 # and exports recorded data as numpy array
class Recorder:

  def __init__(self, thresholdLevel):
    self.threshold = thresholdLevel
    self.acquiredRecordingQueue = None 
  def isAudioLevelAboveThreshold(self):
    pass
  # records current data into numpy array and put it into acquiredRecordingQueue
  def runAcquisition(self):
    pass
  # returns if acquiredRecordingQueue is not empty
  def isDataAvailable(self):
    pass
  # exports and delete acquiesced recording from acquiredRecording Queue
  # as numpy array, 
  def exportRecording(self):
    pass