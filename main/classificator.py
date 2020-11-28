from abc import ABC, abstractclassmethod

class ClassificatorInterface(ABC):
  @abstractclassmethod
  def likelyhood(self, extractedFeatures):
    pass

  @abstractclassmethod
  def train(self, extractedFeaturesList):
    pass

class GMM(ClassificatorInterface):
  pass

class DTW(ClassificatorInterface):
  pass

# EOF