import keyboard
import numpy as np
import math
import queue
import sounddevice as sd
import logging

# Recorder is listening for commands, starts/stop aquairing them
# and exports recorded data as numpy array
class Recorder:

  def __init__(self, thresholdLevel):
    self.sampleRate = 44100
    self.threshold = thresholdLevel
    self.bufferSize = 256
    self.channels = 1
    self.bufferEmptyEndingRecordingLimit = 40
    self.currentBufferNumber = 0
    
    sd.default.samplerate = self.sampleRate
    sd.default.channels = self.channels
    
    self.bufferQueue = queue.Queue()
    self.acquiredBuffersQueue = queue.Queue()
    self.acquiredRecordingQueue = queue.Queue() 

  def exportRecording(self):
    try:
      return self.acquiredRecordingQueue.get_nowait()
    except queue.Empty:
      return None
  
  def isDataAvailable(self):
    return not self.acquiredRecordingQueue.empty()
  
  def run(self):
    recordingStream = sd.InputStream(self.sampleRate, blocksize=self.bufferSize, dtype=('float32'))
    recordingStream.start()
    while True:
      if keyboard.is_pressed('q'):
        break
      if recordingStream.read_available >= self.bufferSize:
        buffer = recordingStream.read(self.bufferSize)[0].flatten()
        self.updateRecorder(buffer)
    recordingStream.stop()
    
  def updateRecorder(self, buffer):
    if buffer is not None:
      if self.isBufferLevelAboveThreshold(buffer):
        self.acquiredBuffersQueue.put(buffer)
        self.currentBufferNumber = 0
      elif not self.acquiredBuffersQueue.empty() and self.currentBufferNumber == self.bufferEmptyEndingRecordingLimit:
        data = np.array([])  
        while not self.acquiredBuffersQueue.empty():
          data = np.concatenate((data, self.acquiredBuffersQueue.get_nowait()))
        self.acquiredRecordingQueue.put(data)
        self.currentBufferNumber = 0
      elif not self.acquiredBuffersQueue.empty():
        self.acquiredBuffersQueue.put(buffer)
        self.currentBufferNumber += 1
  
  # check if given audio buffer exceeds threshold 
  def isBufferLevelAboveThreshold(self, buffer):
    rmsValue = math.sqrt(np.sum(buffer * buffer) / self.bufferSize)
    return rmsValue >= self.threshold

  

  #EOF