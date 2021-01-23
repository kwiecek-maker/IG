from abc import ABC, abstractclassmethod
import random
import numpy as np
import logging
import pandas as pd

# classificator creates function, that approximates
# histogram made out of training data.
class ClassificatorInterface(ABC):
  # Returns likely hood how well function
  # inside classificator approximates histogram of given data
  @abstractclassmethod
  def likelihood(self, extractedFeatures):
    pass

  # creates function that approximates histogram of given data List
  @abstractclassmethod
  def train(self, X):
    pass


# Gaussian Mixture Model
# |n_components| [int] - number of gauss kernels used to estimate
# histogram of the trained data.
# |max_iterations| [int] - number of iterations to perform
# |eps| [float] - what error of estimation should be acceptable
class GMM(ClassificatorInterface):

  def __init__(self, n_components, max_iterations, eps=1e-8):
    self.n_components = n_components
    self.max_iterations = max_iterations
    self.eps = eps

  def __str__(self):
    return self.__repr__() + "\t" + self._arrayToStr(self.means) + "\t" + self._arrayToStr(self.variances) + "\t" + self._arrayToStr(self.weights)

  def __repr__(self):
    return "GMMClassificator"

  def likelihood(self, extractedMfcc: np.array):
    likelihood = []
    for n in range(self.n_components):
      likelihood.append(self.pdf(extractedMfcc, self.means[n], np.sqrt(self.variances[n])))
    likelihood = np.array(likelihood)

    max_likelihoods = np.zeros(likelihood.shape[1])
    for sample in range(likelihood.shape[1]):
      max_likelihoods[sample] = np.max(likelihood[:, sample])

    return np.max(max_likelihoods) - np.mean(max_likelihoods)

  def train(self, extractedMfccList):

    X = np.array([])
    for mfccArray in extractedMfccList:
      X = np.append(X, mfccArray)

    np.random.shuffle(X)
    self.weights = np.ones((self.n_components)) / self.n_components
    self.means = np.random.choice(X, self.n_components)
    self.variances = np.random.random_sample(size=self.n_components)

    for iteration in range(self.max_iterations):

      likelihood = list()
      for n in range(self.n_components):
        likelihood.append(self.pdf(X, self.means[n], np.sqrt(self.variances[n])))
      likelihood = np.array(likelihood)

      b = []
      # Maximization step
      for n in range(self.n_components):
        b.append((likelihood[n] * self.weights[n]) / (
          np.sum([likelihood[i] * self.weights[i]
                  for i in range(self.n_components)], axis=0) + self.eps))

        # Update mean and variance
        self.means[n] = np.sum(b[n] * X) / (np.sum(b[n] + self.eps))
        self.variances[n] = np.sum(b[n] * np.square(X - self.means[n])) / (np.sum(b[n] + self.eps))

        # Update the weights
        self.weights[n] = np.mean(b[n])

    return True

  def pdf(self, data, mean: float, variance: float):
    # A normal continuous random variable.
    s1 = 1/(np.sqrt(2*np.pi*variance))
    s2 = np.exp(-(np.square(data - mean)/(2*variance)))
    return s1 * s2

  def getProbabilityAtValue(self, value: float):
    probability = 0
    for n in range(self.n_components):
      probability += self.pdf(value, self.means[n], self.variances[n])
    return probability

  def getMaxProbabilityFromPDF(self, minValue: float, maxValue: float):
    t = np.arange(minValue, maxValue, self.eps * 1e5)
    y = []
    for time in t:
      y.append(self.getProbabilityAtValue(time))
    return np.max(y)

  @staticmethod
  def _getMiddleValues(inputArray):
    output = [None] * (len(inputArray)-1)
    for index in range(len(inputArray)-1):
      output[index] = (inputArray[index] + inputArray[index+1]) / 2
    return output

  @staticmethod
  def _arrayToStr(array):
    output = "["
    for value in array:
      output += str(value) + ", "
    output = output[:-2] + ']'
    return output

class DTW(ClassificatorInterface):
  def __init__(self):
    self.acquiredMFCC = np.array([])
    self.referenceMFCCMatrix = np.array([])

  def likelihood(self, extractedFeatures):
    return 1

  def train(self, extractedFeaturesList):
    for mfccIndex in range(len(extractedFeaturesList)):
      logging.info(extractedFeaturesList[mfccIndex].shape)

# This classificator is used for testing program only
class FakeClassificator(ClassificatorInterface):
  def __init__(self):
    self.trained = False

  def likelihood(self, extractedFeatures):
    return 20*np.log10(random.uniform(1, 1000))

  def train(self, X):
    return True


# EQ