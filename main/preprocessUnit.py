# Prepares data for futher calculations
import numpy as np
from scipy import signal


class PreprocessUnit:
    def __init__(self, desiredLoudnessLevel=1.0, downsamplingFrequency=8e3, samplingFrequency=44100, onset=0.01,
                 offset=0.01, coefficient=0.95, frameLength=25, overlap=0.5):
        self.desiredLoudnessLevel = desiredLoudnessLevel
        self.downsamplingFrequency = downsamplingFrequency
        self.samplingFrequency = samplingFrequency
        self.onset = onset
        self.offset = offset
        self.coefficient = coefficient
        self.frameLength = frameLength
        self.overlap = overlap

    # Multiplying beginning and end of the audio data with cos ramp
    def cosineRamp(self, inputArrayRecording):
        nOnset = int(self.onset * self.samplingFrequency)
        nOffset = int(self.offset * self.samplingFrequency)

        raisedSignal = np.zeros(nOnset)
        muteSignal = np.zeros(nOffset)

        for n in range(nOnset):
            intensityNum = 0.5 - 0.5 * np.cos(np.pi * n / (nOnset - 1))
            raisedSignal[n] = intensityNum * inputArrayRecording[:nOnset][n]

        for n in range(nOffset):
            intensityNum = 0.5 + 0.5 * np.cos(np.pi * n / (nOffset - 1))
            muteSignal[n] = intensityNum * inputArrayRecording[-nOffset:][n]

        inputArrayRecording[:nOnset] = raisedSignal
        inputArrayRecording[-nOffset:] = muteSignal

        return np.array(inputArrayRecording)

    # Deletes constant component from the recording
    def deleteAverage(self, inputArrayRecording):
      average = np.mean(inputArrayRecording)
      inputArrayRecordingWithoutAverage = inputArrayRecording - average

      return inputArrayRecordingWithoutAverage

    # Normalize given recording to the self.desiredLoudnessLevel
    def normalize(self, inputArrayRecording):
        return np.multiply(self.desiredLoudnessLevel, inputArrayRecording)

    # Preemphasis filter
    def preemphase(self, inputArrayRecording):
        emphasizedSignal = np.append(inputArrayRecording[0], inputArrayRecording[1:] - self.coefficient * inputArrayRecording[:-1])

        return emphasizedSignal

    # Downsample recording to the self.downsamlingFrequency
    def downsample(self, inputArrayRecording):
        numberOfSamples = round(len(inputArrayRecording)*(self.downsamplingFrequency/self.samplingFrequency))

        return signal.resample(inputArrayRecording, numberOfSamples)

    # Split signal into frames (windowing), frameLength = 25 (25 ms), overlap = 0.5 (50%)
    def segmentation(self, inputArrayRecording):
        samplesFrame = round((self.frameLength*self.downsamplingFrequency)/1000)
        samplesOverlap = round(samplesFrame*self.overlap)
        hamming = np.hamming(samplesFrame)

        chunk = range(0, len(inputArrayRecording)-samplesOverlap, samplesOverlap)

        framesAmmount = 0
        for idx, k in enumerate(chunk):
            if len(inputArrayRecording[k:k+samplesFrame]) == samplesFrame:
                framesAmmount += 1

        segmentArray = np.zeros(shape=(samplesFrame, framesAmmount))

        for idx, i in enumerate(chunk):
            currentArray = inputArrayRecording[i:i+samplesFrame]
            if len(currentArray) == samplesFrame:
                segmentArray[:, idx] = hamming*currentArray

        return segmentArray

    # Performs all required processing using methods in this class for input array of audio samples
    def process(self, inputArray):
        data = self.cosineRamp(list(inputArray))
        data = self.deleteAverage(data)
        data = self.normalize(data)
        data = self.preemphase(data)
        data = self.downsample(data)
        matrixOfArrays = self.segmentation(data)

        return matrixOfArrays

#EOF