import numpy as np
import scipy.signal
import os


class LPCcoder:
    def __init__(self, predictionLength, frameSize):
        self.predictionLength = predictionLength
        self.frameSize = frameSize

    def __repr__(self):
        return "LPC coder. prediction Length: %d, BLock size: %d" % (self.predictionLength, self.frameSize)
    
    def __str__(self):
        return self.__repr__()

    def codeRecording(self, inputRecording, fs):
        signalLength = np.size(inputRecording)
        frameSamples = np.int(np.floor(signalLength * self.frameSize))
        numOfBlocks = np.int(np.floor(signalLength / frameSamples))

        predictionCoefficientMemory = np.zeros((numOfBlocks, self.predictionLength))
        predictionError = np.zeros(signalLength)

        # Memory state of prediction filter
        stateOfPredictionFilter = np.zeros(self.predictionLength)

        for i in range(0, numOfBlocks):
            # Temporary matrix, which contain x rates of filter for each sample in frame
            A = np.zeros((frameSamples - self.predictionLength, self.predictionLength))

            for j in range(0, frameSamples - self.predictionLength):
                A[j, :] = np.flipud(inputRecording[i * frameSamples + j + np.arange(self.predictionLength)])

            # Construct our desired target signal d, one sample into the future
            d = inputRecording[i * frameSamples + np.arange(self.predictionLength, frameSamples)]
            # Compute the prediction filter
            predictionCoefficientMemory[i, :] = np.dot(np.dot(np.linalg.pinv(np.dot(A.transpose(), A)), A.transpose()), d)
            hperr = np.hstack([1, -predictionCoefficientMemory[i, :]])
            predictionError[i * frameSamples + np.arange(0, frameSamples)], stateOfPredictionFilter = scipy.signal.lfilter(hperr, [1], inputRecording[i * frameSamples + np.arange(0, frameSamples)], zi=stateOfPredictionFilter)

        return predictionCoefficientMemory, predictionError, inputRecording.shape, frameSamples, numOfBlocks

    # Exports calculated LPC coefficients into txt files in speechGenSrc/dataBase/
    @staticmethod
    def exportLPC(recordingName, predictionCoefficientMemory, predictionError):
        with open(os.getcwd() + "\\SpeechGenSrc\\database\\" + recordingName + "predictionCoefficientMemory.txt", 'w+') as f:
            f.writelines(predictionCoefficientMemory)

        with open(os.getcwd() + "\\SpeechGenSrc\\database\\" + recordingName + "predictionError.txt", 'w+') as f:
            f.writelines(predictionError)