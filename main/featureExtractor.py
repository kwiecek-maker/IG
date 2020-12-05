from abc import ABC, abstractmethod
from scipy.signal import hanning
import numpy as np

class FeatureExtractorInterface(ABC):

    def __init__(self, soundframesArray):
        self.soundArray = soundframesArray
        self.frames_n = self.soundArray.shape[1]
        self.frame_len = self.soundArray.shape[0]

    @abstractmethod
    def exctract(self):
        return


class MFCC(FeatureExtractorInterface):

    def __init__(self, soundframesArray, samplerate=44100, nceps=13, nfilt=26, nfft=512):
        super(MFCC, self).__init__(soundframesArray)
        self.nceps = nceps
        self.nfilt = nfilt
        self.nfft = nfft
        self.samplerate = samplerate

    def _fft(self):
        w = hanning(self.frame_len)
        self.spectrumArray = np.zeros((self.nfft, self.frames_n))
        for i in range(self.frames_n):
            for k in range(self.nfft):
                self.spectrumArray[k, i] = (1/self.frame_len)*np.abs(np.sum(self.soundArray[:, i]*w*np.exp((-1j*2*np.pi*k*np.arange(self.frame_len))/self.frame_len)))**2
        self.spectrumArray = self.spectrumArray[:self.nfft//2+1, :]

    def _freq2mel(self, freq):
        return 1125*np.log(1 + freq/700)

    def _mel2freq(self, mel):
        return 700*(np.exp(mel/1125)-1)

    def _freq2binfft(self, freq):
        return np.floor((self.nfft+1)*freq/self.samplerate)

    def _melfbank(self):
        freq_low = 0  # lower frequency
        freq_up = self.samplerate//2  # upper frequency

        mel_low = self._freq2mel(freq_low)
        mel_up = self._freq2mel(freq_up)
        # liner values in mel domain
        melArray = np.linspace(mel_low, mel_up, self.nfilt+2)
        # convert mel to Hz domain
        freqArray = np.array([self._mel2freq(mel) for mel in melArray])
        # convert Hz to fft bin
        freqbinArray = np.array([self._freq2binfft(freq) for freq in freqArray])

        # calculate mel filterbank (każdy filtr ma po 257 wartości)
        self.melbank = np.zeros((self.nfft//2+1, self.nfilt))

        for m in range(1, self.nfilt+1):
            fm_previous = freqbinArray[m-1]
            fm = freqbinArray[m]
            fm_next = freqbinArray[m+1]

            # left slope
            for k in range(fm_previous, fm+1):
                self.melbank[k, m] = (k-fm_previous)/(fm-fm_previous)
            # right slope
            for k in range(fm, fm_next+1):
                self.melbank[k, m] = (fm_next-k)/(fm_next-fm)

    def _applymelfbank(self):
        # for each frame we achieved 26 ceps
        self.filteredArray = np.dot(self.spectrumArray.T, self.melbank).T

    def _logfbank(self):
        self.logfilteredArray = np.log10(self.filteredArray)

    def _dct(self):
        self.mfcc_cepstras = np.zeros((self.nceps, self.frames_n))
        for i in range(self.frames_n):
            for j in range(self.nceps):
                self.mfcc_cepstras[j, i] = np.sum(self.logfilteredArray[:, i]*np.cos(j*np.arange(0.5, self.nfilt+0.5)*np.pi/self.nfilt))

    def deltas(self):
        pass

    def deltas_deltas(self):
        pass

    def exctract(self):
        # converting signal frames to frequency domain
        self._fft()
        # computing the Mel filterbank
        self._melfbank()
        # apply Mel filterbank to signal's spectrum
        self._applymelfbank()
        # logarithm
        self._logfbank()
        # decorelating filterbank coefs - caltulating mfcc cepstras
        self._dct()


class HFCC(FeatureExtractorInterface):

    def extract(self, ):
        pass

# EOF
# def f0(self,):
#   pass
#
# def formants(self,):
#   pass
