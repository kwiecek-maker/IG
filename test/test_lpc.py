import os
import soundfile
import matplotlib.pyplot as plt
from SpeechGenSrc.LPCcoder import LPCcoder
from SpeechGenSrc.decode import Decoder


def main():
    filePath = os.path.join(os.getcwd(), 'wylacz.wav')
    inputRecording, fs = soundfile.read(filePath)
    lpc = LPCcoder(predictionLength=10, frameSize=0.025)
    predictionCoefficientMemory, predictionError, inputRecordingShape, frameSamples, numOfBlocks = lpc.codeRecording(inputRecording, fs)

    decoder = Decoder(LPCmatrix=predictionCoefficientMemory, LPCerror=predictionError, inputRecordingShape=inputRecordingShape,
                      frameSamples=frameSamples, numOfBlocks=numOfBlocks, fs=fs, predictionLength=10)

    synthesis = decoder.decodeRecording()
    outputName = 'synthesis_command.wav'
    decoder.generateRecording(outputPath=os.path.join(os.getcwd(), outputName), synthesisRecording=synthesis)
    decoder.playRecording(wavFilePath=os.path.join(os.getcwd(), outputName))

    # Plot signal and it's prediction error:
    plt.figure(figsize=(10, 8))
    plt.plot(inputRecording)
    plt.plot(predictionError, 'r')
    plt.xlabel('Sample')
    plt.ylabel('Normalized Value')
    plt.legend(('Original', 'Prediction Error'))
    plt.title('LPC Coding')
    plt.grid()
    plt.show()

    # Plot original and predicted
    plt.figure(figsize=(10, 8))
    plt.plot(inputRecording)
    plt.plot(synthesis)
    plt.xlabel('Sample')
    plt.ylabel('Normalized Value')
    plt.legend(('Original', 'Prediction'))
    plt.title('LPC Decoding')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()