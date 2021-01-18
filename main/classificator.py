from abc import ABC, abstractclassmethod
import random

# classificator creates function, that approximates
# histogram made out of training data.
class ClassificatorInterface(ABC):
  # Returns likely hood how well function
  # inside classificator approximates histogram of given data
  @abstractclassmethod
  def likelyhood(self, extractedFeatures):
    pass

  # creates function that approximates histogram of given data List
  @abstractclassmethod
  def train(self, extractedFeaturesList):
    pass

# GMM creates gaussian mixture that approximates given data
# |nComponents| - The number of mixture components.
# |tolerance| - The convergence threshold. EM iterations will stop when the
# lower bound average gain is below this threshold.
# |maxIteration| - The number of EM iterations to perform.
class GMM(ClassificatorInterface):
  def __init__(self, nComponents=1, tolerance=1e-3, maxIterations=100 ):
    self.nComponents = nComponents
    self.tolerance = tolerance
    self.maxIterations = maxIterations

  def likelyhood(self, extractedFeatures):
    pass

  def train(self, extractedFeaturesList):
    pass

# class DTW(ClassificatorInterface):
#   def __init__(self, extractedFeatures):
#     pass

#   def likelyhood(self, extractedFeatures):
#     pass

#   def train(self, extractedFeaturesList):
#     pass

# This classificator is used for testing program only
class FakeClassificator(ClassificatorInterface):
  def __init__(self):
    self.trained = False

  def likelyhood(self, extractedFeatures):
    return random.uniform(1, 1000)

  def train(self, extractedFeatures):
    return True


# EQ