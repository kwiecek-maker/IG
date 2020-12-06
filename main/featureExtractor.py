from abc import ABC, abstractmethod
import numpy as np
import itertools

class FeatureExtractorInterface(ABC):

    def __init__(self, matrixOfSegments):
        self.matrixOfSegments = matrixOfSegments
        self.numberOfSegments = self.matrixOfSegments.shape[1]
        self.lengthOfSegment = self.matrixOfSegments.shape[0]

    @abstractmethod
    def exctract(self):
        return


class MFCC(FeatureExtractorInterface):

    def __init__(self, matrixOfSegments, samplerate=44100, numberOfCepstras=13,
                 numberOfMelFilters=26, numberOfFrequencyBins=512):
        super(MFCC, self).__init__(matrixOfSegments)
        self.samplerate = samplerate
        self.numberOfCepstras = numberOfCepstras
        self.numberOfMelFilters = numberOfMelFilters
        self.numberOfFrequencyBins = numberOfFrequencyBins

    def fft(self):
        self.spectrumArray = np.zeros((self.numberOfFrequencyBins, self.numberOfSegments))
        alpha = 1/self.numberOfFrequencyBins
        for i in range(self.numberOfSegments):
            iSegment = self.matrixOfSegments[:, i]
            # zero-padding
            if self.numberOfFrequencyBins > self.lengthOfSegment:
                iSegment = np.append(iSegment, np.zeros(self.numberOfFrequencyBins-self.lengthOfSegment))
            for k in range(self.numberOfFrequencyBins):
                fourierKernel = np.exp(
                        (-1j*2*np.pi*k*np.arange(self.numberOfFrequencyBins))/self.numberOfFrequencyBins)
                self.spectrumArray[k, i] = alpha*np.abs(np.sum(iSegment*fourierKernel))**2
        self.spectrumArray = self.spectrumArray[:self.numberOfFrequencyBins//2+1, :]

    def freq2mel(self, freq):
        return 1125*np.log(1 + freq/700)

    def mel2freq(self, mel):
        return 700*(np.exp(mel/1125)-1)

    def freq2binfft(self, freq):
        return np.floor((self.numberOfFrequencyBins+1)*freq/self.samplerate)

    def melfbank(self):

        lowFrequency = 0
        highFrequency = self.samplerate//2

        lowMelFreqency = self.freq2mel(lowFrequency)
        highMelFrequency = self.freq2mel(highFrequency)

        melFrequencyArray = np.linspace(lowMelFreqency, highMelFrequency, self.numberOfMelFilters+2)
        frequencyArray = np.array([self.mel2freq(mel) for mel in melFrequencyArray])
        frequencyBinArray = np.array([self.freq2binfft(freq) for freq in frequencyArray])

        # calculate mel filterbank
        self.melbank = np.zeros((self.numberOfFrequencyBins//2+1, self.numberOfMelFilters))
        for m in range(1, self.numberOfMelFilters+1):
            previousFrequencyBin = frequencyBinArray[m-1]
            currentFrequencyBin = frequencyBinArray[m]
            nextFrequencyBin = frequencyBinArray[m+1]

            # left slope
            for k in range(previousFrequencyBin, currentFrequencyBin+1):
                self.melbank[k, m] = (k-previousFrequencyBin)/(currentFrequencyBin-previousFrequencyBin)
            # right slope
            for k in range(currentFrequencyBin, nextFrequencyBin+1):
                self.melbank[k, m] = (nextFrequencyBin-k)/(nextFrequencyBin-currentFrequencyBin)

    def applymelfbank(self):
        # for each segment achieve numberOfCepstras cepstras
        self.filteredArray = np.dot(self.spectrumArray.T, self.melbank).T

    def logfbank(self):
        self.logfilteredArray = np.log10(self.filteredArray)

    def dct(self):
        self.mfccFeaturesArray = np.zeros((self.numberOfCepstras, self.numberOfSegments))
        for i, j in itertools.product(range(self.numberOfSegments), range(self.numberOfCepstras)):
            self.mfccFeaturesArray[j, i] = np.sum(
                self.logfilteredArray[:, i]*np.cos(j*np.arange(0.5, self.numberOfMelFilters+0.5)*np.pi/self.numberOfMelFilters))

    def exctract(self):
        self.fft()
        self.melfbank()
        self.applymelfbank()
        self.logfbank()
        self.dct()
        return self.mfccFeaturesArray


class HFCC(FeatureExtractorInterface):

    def extract(self, ):
        pass

# EOF
# def f0(self,):
#   pass
#
# def formants(self,):
#   pass
