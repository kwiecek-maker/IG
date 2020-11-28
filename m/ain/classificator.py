class ClassificatorInterface(ABC):
  @abstractmethod
  def likelyhood(self, extractedFeatures):
    pass

  @abstractmethod
  def train(self, extractedFeaturesList):
    pass

class GMM(ClassificatorInterface):
  pass

class DTW(ClassificatorInterface):
  pass