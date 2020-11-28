from abc import ABC, abstractmethod
class FeatureExtractorInterface(ABC):
  def f0():
    pass

  def formants():
    pass

  @abstractmethod
  def exctract(soundArray):
    return 

class MFCC(FeatureExtractorInterface):
  def fft(soundArray):
    pass
  def dcts(SoundarrayAfterFft): 
    pass
  def frame(inputarray):
    pass
  def exctract(soundArray):
    pass

class HFCC(FeatureExtractorInterface):
  def extract():
    pass