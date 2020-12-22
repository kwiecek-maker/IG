import numpy as np
import os

class LPCcoder:

    def __init__(self, predictionLength, blockSize):
        self.predictionLength = predictionLength
        self.blockSize = blockSize

    def __repr__(self):
        return "LPC coder. prediction Length: %d, BLock size: %d" % (self.predictionLength, self.blockSize)
    
    def __str__(self):
        return self.__repr__()

    def codeRrcording(self, inputRecording):
        signalLength = np.size(inputRecording)
        numOfBlocks = np.int(np.floor(signalLength / self.blockSize))
        
        predictionCoefficientMemory = np.zeros((numOfBlocks, self.predictionLength))
        predictionError = np.zeros(signalLength)

        return [predictionCoefficientMemory, predictionError]

    # Exports calculated LPC coefficients into txt files in speechGenSrc/dataBase/
    def exportLPC(self, recordingName, predictionCoefficientMemory, predicionError):
        with open(os.getcwd() + "\\SpeechGenSrc\\database\\" + recordingName + "predictionCoefficientMemory.txt", 'w+') as f:
            f.writelines("predictionCoefficientMemory")

        with open(os.getcwd() + "\\SpeechGenSrc\\database\\" + recordingName + "predictionError.txt", 'w+') as f:
            f.writelines("predictionError")