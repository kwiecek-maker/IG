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
                 numberOfMelFilters=26, numberOfFrequencyBins=512, appendFrameEnergy=True):
        super(MFCC, self).__init__(matrixOfSegments)
        self.sampleRate = samplerate
        self.numberOfCepstras = numberOfCepstras
        self.numberOfMelFilters = numberOfMelFilters
        self.numberOfFrequencyBins = numberOfFrequencyBins
        self.appendFrameEnergy = appendFrameEnergy

        self.spectrumArray = np.zeros((self.numberOfFrequencyBins, self.numberOfSegments))
        self.melBank = np.zeros((self.numberOfFrequencyBins // 2 + 1, self.numberOfMelFilters))
        self.mfccFeaturesArray = np.zeros((self.numberOfCepstras, self.numberOfSegments))

    def fft(self):
        alpha = 1/self.numberOfFrequencyBins

        # truncate segment
        if self.numberOfFrequencyBins < self.lengthOfSegment:
            self.lengthOfSegment = self.numberOfFrequencyBins

        for i in range(self.numberOfSegments):
            segment_i = self.matrixOfSegments[:self.lengthOfSegment, i]

            # zero-padding
            if self.numberOfFrequencyBins > self.lengthOfSegment:
                segment_i = np.append(segment_i, np.zeros(self.numberOfFrequencyBins-self.lengthOfSegment))

            for k in range(self.numberOfFrequencyBins):
                fourier_kernel = np.exp(
                        (-1j*2*np.pi*k*np.arange(self.numberOfFrequencyBins))/self.numberOfFrequencyBins)
                self.spectrumArray[k, i] = alpha*np.abs(np.sum(segment_i*fourier_kernel))**2

        self.spectrumArray = self.spectrumArray[:self.numberOfFrequencyBins//2+1, :]

    @staticmethod
    def freq2mel(freq):
        return 1127*np.log(1 + freq/700)

    @staticmethod
    def mel2freq(mel):
        return 700*(np.exp(mel/1127)-1)

    def freq2fft_bin(self, freq):
        return np.floor((self.numberOfFrequencyBins+1)*freq/self.sampleRate)

    def mel_bank(self):
        low_frequency = 0
        high_frequency = self.sampleRate//2
        low_mel_frequency = self.freq2mel(low_frequency)
        high_mel_frequency = self.freq2mel(high_frequency)
        mel_frequency_array = np.linspace(low_mel_frequency, high_mel_frequency, self.numberOfMelFilters+2)
        frequency_array = np.array([self.mel2freq(mel) for mel in mel_frequency_array])
        frequency_bins_array = np.array([self.freq2fft_bin(freq) for freq in frequency_array])

        # calculate mel filter bank
        for m in range(1, self.numberOfMelFilters+1):
            previous_frequency_bin = frequency_bins_array[m-1]
            current_frequency_bin = frequency_bins_array[m]
            next_frequency_bin = frequency_bins_array[m+1]
            # left slope
            for k in range(int(previous_frequency_bin), int(current_frequency_bin)+1):
                self.melBank[k, m-1] = (k-previous_frequency_bin)/(current_frequency_bin-previous_frequency_bin)
            # right slope
            for k in range(int(current_frequency_bin), int(next_frequency_bin)+1):
                self.melBank[k, m-1] = (next_frequency_bin-k)/(next_frequency_bin-current_frequency_bin)

    def apply_mel_bank(self):
        self.filteredArray = np.dot(self.spectrumArray.T, self.melBank).T

    def logfbank(self):
        self.logfilteredArray = np.log(self.filteredArray)

    def dct(self):
        for i, j in itertools.product(range(self.numberOfSegments), range(self.numberOfCepstras)):
            self.mfccFeaturesArray[j, i] = 2*np.sum(self.logfilteredArray[:, i] *
                            np.cos(np.pi * j * np.arange(0.5, self.numberOfMelFilters+0.5)/self.numberOfMelFilters))
            # scaling factor
            if j == 0:
                self.mfccFeaturesArray[j, i] *= np.sqrt(1/(4*self.numberOfMelFilters))
            else:
                self.mfccFeaturesArray[j, i] *= np.sqrt(1/(2*self.numberOfMelFilters))

    # transform cepstrum 0 -> total spectral energy of frame
    def spectral_energy(self):
        for frame in range(self.numberOfSegments):
            self.mfccFeaturesArray[0, frame] = np.log(np.sum(np.abs(self.spectrumArray[:, frame])))

    def exctract(self):
        self.fft()
        self.mel_bank()
        self.apply_mel_bank()
        self.logfbank()
        self.dct()
        if self.appendFrameEnergy:
            self.spectral_energy()

        return self.mfccFeaturesArray


# class HFCC(FeatureExtractorInterface):
#
#     def extract(self, ):
#         pass

# EOF
# def f0(self,):
#   pass
#
# def formants(self,):
#   pass
