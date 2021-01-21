"""
this file should be run with addictional command arugment. For example:
$ python decode.py "command"

It will automaticly play decoded signal.
"""

import numpy as np
import scipy.signal
import soundfile
import simpleaudio


# change given name to the associated LPC coefficient file and LPC coefficient error
def transposeName(name):
    return ["", ""]


# walks through directories and read files associated with
# given filename. Returns None when files are not found.
def readTxtfiles(filename):
    return None


# decode txt files from database files
class Decoder:

    def __init__(self, LPCmatrix, LPCerror, inputRecordingShape, frameSamples, numOfBlocks, fs, predictionLength):
        self.LPCmatrix = LPCmatrix
        self.LPCerror = LPCerror
        self.inputRecordingShape = inputRecordingShape
        self.frameSamples = frameSamples
        self.numOfBlocks = numOfBlocks
        self.fs = fs

        self.predictionLength = predictionLength

    # Decoder
    def decodeRecording(self):
        # Initialize reconstructed signal memory
        synthesisRecording = np.zeros(self.inputRecordingShape)
        # Initialize Memory state of prediction filter
        state = np.zeros(self.predictionLength)

        for i in range(0, self.numOfBlocks):
            # Predictive reconstruction filter: hperr from numerator to denominator
            hperr = np.hstack([1, -self.LPCmatrix[i, :]])
            synthesisRecording[i * self.frameSamples + np.arange(0, self.frameSamples)], state = scipy.signal.lfilter([1], hperr, self.LPCerror[i * self.frameSamples + np.arange(0, self.frameSamples)], zi=state)

        return synthesisRecording

    def generateRecording(self, outputPath, synthesisRecording):
        soundfile.write(outputPath, synthesisRecording, self.fs)

    @staticmethod
    def playRecording(wavFilePath):
        waveObj = simpleaudio.WaveObject.from_wave_file(wavFilePath)
        playObj = waveObj.play()

        while True:
            if playObj.is_playing():
                print('Playing...')
            else:
                print('Ended')
                break


if __name__ == "__main__":
    pass
    # name = sys.argv[1]
    # [LPCmatrixName, LPCerrrorName] = transposeName(name)
    #
    # LPCMatrix = readTxtfiles(LPCmatrixName)
    # LPCError = readTxtfiles(LPCerrrorName)

    # if LPCMatrix is not None and LPCError is not None:
    #     decoder = Decoder(LPCMatrix, LPCError)
    #     decoder.decodeRecording()
    #     decoder.generateRecording()
