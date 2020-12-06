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
    def cosineRamp(self, inputArraySignal):
        nOnset = int(self.onset * self.samplingFrequency)
        nOffset = int(self.offset * self.samplingFrequency)

        raisedSignal = np.zeros(nOnset)
        muteSignal = np.zeros(nOffset)

        for n in range(nOnset):
            intensityNum = 0.5 - 0.5 * np.cos(np.pi * n / (nOnset - 1))
            raisedSignal[n] = intensityNum * inputArraySignal[:nOnset][n]

        for n in range(nOffset):
            intensityNum = 0.5 + 0.5 * np.cos(np.pi * n / (nOffset - 1))
            muteSignal[n] = intensityNum * inputArraySignal[-nOffset:][n]

        inputArraySignal[:nOnset] = raisedSignal
        inputArraySignal[-nOffset:] = muteSignal

        return np.array(inputArraySignal)

    # Deletes constant component from the data
    def deleteAverage(self, inputArraySignal):
        average = np.mean(inputArraySignal)
        inputArraySignalWithoutAverage = inputArraySignal - average

        return inputArraySignalWithoutAverage

    # Normalize given data to the self.desiredLoudnessLevel
    def normalize(self, inputArraySignal):
        inputArraySignalNormalize = np.multiply(self.desiredLoudnessLevel, inputArraySignal)

        return inputArraySignalNormalize

    # Preemphasis filter
    def preemphase(self, inputArraySignal):
        emphasizedSignal = np.append(inputArraySignal[0], inputArraySignal[1:] - self.coefficient * inputArraySignal[:-1])

        return emphasizedSignal

    # Downsample data to the self.downsamlingFrequency
    def downsample(self, inputArraySignal):
        downsamplingFactor = int(round(self.samplingFrequency / self.downsamplingFrequency))
        inputArraySignalDecimate = signal.decimate(inputArraySignal, q=downsamplingFactor, ftype='fir')

        return inputArraySignalDecimate

    # Split signal into frames/segments, frameLength = 25 (25 ms), overlap = 0.5 (50%)
    # Return matrix of arrays - each column is a 25 ms segment
    def segmentation(self, inputArraySignal):
        samplesFrame = round((self.frameLength * self.downsamplingFrequency) / 1000)
        samplesOverlap = round(samplesFrame * self.overlap)
        hamming = np.hamming(samplesFrame)

        chunk = range(0, len(inputArraySignal) - samplesOverlap, samplesOverlap)

        framesAmmount = 0
        for idx, k in enumerate(chunk):
            if len(inputArraySignal[k:k + samplesFrame]) == samplesFrame:
                framesAmmount += 1

        matrixOfSegments = np.zeros(shape=(samplesFrame, framesAmmount))

        for idx, i in enumerate(chunk):
            currentArray = inputArraySignal[i:i + samplesFrame]
            if len(currentArray) == samplesFrame:
                matrixOfSegments[:, idx] = hamming * currentArray

        return matrixOfSegments

    # Performs all required processing using methods in this class for input array of audio samples
    def process(self, inputArray):
        data = self.cosineRamp(list(inputArray))
        data = self.deleteAverage(data)
        data = self.normalize(data)
        data = self.preemphase(data)
        data = self.downsample(data)
        matrixOfSegments = self.segmentation(data)

        return matrixOfSegments

# EOF
