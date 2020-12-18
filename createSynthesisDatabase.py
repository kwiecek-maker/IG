# This file creates LPC codes and saves them as .txt file for future encoding. 
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as sp

currentDirectory = os.getcwd()

data, samplerate = sf.read(currentDirectory + "\\testRecordings\\testRecording.wav")
data = data[:, 0].flatten()

predictorLength = 12
blockSize = 640

signalLength = np.size(data)

predictionError = np.zeros(signalLength)
numOfBlocks = np.int(np.floor(signalLength / blockSize))
memoryPredictionFilterState = np.zeros(predictorLength)
predictionCoefficientMemory  = np.zeros((numOfBlocks, predictorLength))

for currentBlockNumber in range(numOfBlocks):
  # temporary block has size of blockSize - predictorLength -> trick to avoid zeros in the 
  tempBlock = np.zeros((blockSize - predictorLength, predictorLength)) 
  for sampleInBlock in range(blockSize - predictorLength):
    tempBlock[sampleInBlock, :] = np.flipud(
      data[currentBlockNumber * blockSize + sampleInBlock + np.arange(predictorLength)])
  
  futurePrediction = data[currentBlockNumber * blockSize + np.arange(predictorLength, blockSize)]
  
  # pinv - computes the (Moore-Penrose) pseudo-inverse of a matrix: 
  # wiki: https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
  # yt: https://youtu.be/5bxsxM2UTb4
  predictionCoefficientMemory[currentBlockNumber, :] = np.dot(np.dot(
    np.linalg.pinv(np.dot(tempBlock.transpose(), tempBlock)),
    tempBlock.transpose()), futurePrediction)
                                                              
  predictionCoefficientMemoryError = np.hstack(
    [1, - predictionCoefficientMemory[currentBlockNumber, :]])
  
  predictionError[currentBlockNumber * blockSize + np.arange(blockSize)], memoryPredictionFilterState = sp.lfilter(
    predictionCoefficientMemoryError, [1], data[currentBlockNumber*blockSize + np.arange(0, blockSize)], zi=memoryPredictionFilterState)
  
averageError = np.dot(predictionError, predictionError) / np.max(np.size(predictionError))

predictionError = np.array(predictionError / max(predictionError), dtype='float32')
signalExport = np.zeros((signalLength, 2))
signalExport[:, 0] = predictionError
signalExport[:, 1] = predictionError
# SYNTHESIS
destinationPath = currentDirectory + r"\testRecordings\synthesisedTestRecording.wav"
sf.write(destinationPath, predictionError, samplerate=samplerate)



