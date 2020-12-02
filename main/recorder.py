import argparse
from os import fsdecode
import keyboard
import logging
import numpy as np
import math
import matplotlib.pyplot as plt
import queue
import sounddevice as sd
import sys
import time


# Helper function for argument parsing.
def intOrString(text):
    try:
        return int(text)
    except ValueError:
        return text

# Recorder is listening for commands, starts/stop aquairing them
# and exports recorded data as numpy array
class Recorder:

  def __init__(self, thresholdLevel):
    self.sampleRate = 44100
    self.threshold = thresholdLevel
    self.bufferSize = 256
    self.channels = 2
    
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
    recordingStream = sd.InputStream(self.sampleRate, blocksize=self.bufferSize, device=1)
    recordingStream.start()
    while True:
      if keyboard.is_pressed('q'):
          break
      try:
        buffer = recordingStream.read(self.bufferSize)
        self.updateRecorder(buffer)
      except Exception as e:
        logging.error(type(e).__name__ + ': ' + str(e))
    recordingStream.stop()
    
  def updateRecorder(self, buffer):
    if buffer is not None:
      if self.isBufferLevelAboveThreshold(buffer):
        self.acquiredBuffersQueue.put(buffer)
      elif self.acquiredBuffersQueue.not_empty():
        data = np.array([])
        while self.acquiredRecordingQueue.not_empty():
          np.concatenate(data, self.acquiredBuffersQueue.get_nowait())
        self.acquiredRecordingQueue.put(data)
        
  # check if given audio buffer exceeds threshold 
  def isBufferLevelAboveThreshold(self, buffer):
    rmsValue = math.sqrt(np.sum(buffer[:,1] * buffer[:,1]) / self.bufferSize)
    return rmsValue >= self.threshold

  

  #EOF