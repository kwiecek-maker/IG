import keyboard
import numpy as np
import math
import queue
import sounddevice as sd
import logging
import os
import random
import soundfile as sf
import time
from abc import ABC, abstractclassmethod


class RecorderInterface(ABC):
  @abstractclassmethod
  def isDataAvailable(self):
    return False

  @abstractclassmethod
  def exportRecording(self):
    return False

  @abstractclassmethod
  def run(self):
    return False

# Recorder is listening for commands, starts/stop aquairing them
# and exports recorded data as numpy array
class Recorder(RecorderInterface):

  def __init__(self, thresholdLevel):
    self.sampleRate = 44100
    self.threshold = thresholdLevel
    self.bufferSize = 256
    self.channels = 1
    self.bufferNumberLimitFromExceededThreshold = 40
    self.bufferNumberFromExceededThreshold = 0

    sd.default.samplerate = self.sampleRate
    sd.default.channels = self.channels

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
        self.processBuffer(buffer)
    recordingStream.stop()

  def processBuffer(self, buffer):
    if self.isBufferLevelAboveThreshold(buffer):
      self.acquiredBuffersQueue.put(buffer)
      self.bufferNumberFromExceededThreshold = 0
    elif not self.acquiredBuffersQueue.empty() and self.bufferNumberFromExceededThreshold == self.bufferNumberLimitFromExceededThreshold:
      data = np.array([])
      while not self.acquiredBuffersQueue.empty():
        data = np.concatenate((data, self.acquiredBuffersQueue.get_nowait()))
      self.acquiredRecordingQueue.put(data)
      logging.info("Recording Acquired: length: " + str(float(len(data)) / self.sampleRate)[0:5] + "s")
      self.bufferNumberFromExceededThreshold = 0
    elif not self.acquiredBuffersQueue.empty():
      self.acquiredBuffersQueue.put(buffer)
      self.bufferNumberFromExceededThreshold += 1

  # check if given audio buffer exceeds RMS threshold
  def isBufferLevelAboveThreshold(self, buffer):
    rmsValue = math.sqrt(np.dot(buffer, buffer) / self.bufferSize)
    return rmsValue >= self.threshold

# This fake recorder is used only for testing progream
class FakeRecorder(RecorderInterface):

  # recordingCreationFrequency is telling how often
  def __init__(self, RelativePath, recordingAcquisitionFrequency = 0.5):
    self.recordingProducingPeriod = recordingAcquisitionFrequency ** -1
    self.acquiredRecordingQueue = queue.Queue()

    self.recordingPaths = list()
    for root, dirs, files in os.walk(RelativePath, topdown=False):
      for path in files:
        self.recordingPaths.append(os.path.join(root, path))

  def isDataAvailable(self):
    return not self.acquiredRecordingQueue.empty()

  def exportRecording(self):
    try:
      return self.acquiredRecordingQueue.get_nowait()
    except queue.Empty:
      return None

  def run(self):
    while True:
      if keyboard.is_pressed('q'):
        break
      else:
        path = random.choice(self.recordingPaths)
        data, samplerate = sf.read(path)
        data = self.flattenData(data)
        self.acquiredRecordingQueue.put(data)
      time.sleep(self.recordingProducingPeriod)

  @staticmethod
  def flattenData(data):
        if len(data.shape) >= 2:
            data = data[:, 0].flatten()
        else:
            data = data.flatten()
        return data
  #EOF