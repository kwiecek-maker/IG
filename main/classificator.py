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
  def likelyhood(self, extractedFeatures):
    pass

  # creates function that approximates histogram of given data List
  @abstractclassmethod
  def train(self, X):
    pass

# GMM creates gaussian mixture that approximates given data
# |nComponents| - The number of clusters in which the algorithm
#                 must split the data set
# |max_iter| - The number of iteration that the algorithm will
#              go throw to find the clusters lower bound average
#              gain is below this threshold.
# |comp_names| -  In case it is setted as a list of string it will use to
#              name the clusters
class GMM(ClassificatorInterface):
  def __init__(self, nComponents, max_iter = 100, maxIterations=100, comp_names=None):
    self.n_componets = nComponents
    self.max_iter = max_iter
    if comp_names == None:
        self.comp_names = [f"comp{index}" for index in range(self.n_componets)]
    else:
        self.comp_names = comp_names
    # pi list contains the fraction of the dataset for every cluster
    self.pi = [1/self.n_componets for comp in range(self.n_componets)]

  # Multivariate normal derivation formula,
  # the normal distribution for vectors it requires the following parameters

  # |X| 1-d numpy array - The row-vector for which we want
  # to calculate the distribution

  # |mean_vector| 1-d numpy array - The row-vector that contains
  # the means for each column

  # |covariance_matrix| 2-d numpy array (matrix) - The 2-d matrix that
  # contain the covariances for the features
  def multivariate_normal(self, X, mean_vector, covariance_matrix):
    return (2 * np.pi) ** (
      -len(X) / 2) * np.linalg.det(
        covariance_matrix) ** (-1 / 2) * np.exp(
          - np.dot( np.dot( ( X - mean_vector).T,
                           np.linalg.inv(covariance_matrix)),
                   (X-mean_vector))/2)

  def train(self, X):
    '''
        The function for training the model
            :param X: 2-d numpy array
                The data must be passed to the algorithm as 2-d array,
                where columns are the features and the rows are the samples
    '''
    new_X = np.array_split(X, self.n_componets)
    self.mean_vector = [np.mean(x, axis=0) for x in new_X]
    self.covariance_matrixes = [np.cov(x.T) for x in new_X]
    del new_X

    for iteration in range(self.max_iter):
      self.r = np.zeros((len(X), self.n_componets))
      for n in range(len(X)):
          for k in range(self.n_componets):
              self.r[n][k] = self.pi[k] * self.multivariate_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
              self.r[n][k] /= sum([self.pi[j]*self.multivariate_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)])

      N = np.sum(self.r, axis=0)
      self.mean_vector = np.zeros((self.n_componets, len(X[0])))

      for k in range(self.n_componets):
          for n in range(len(X)):
              self.mean_vector[k] += self.r[n][k] * X[n]

      self.mean_vector = [1/N[k]*self.mean_vector[k] for k in range(self.n_componets)]
      self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_componets)]

      for k in range(self.n_componets):
          self.covariance_matrixes[k] = np.cov(X.T, aweights=(self.r[:, k]), ddof=0)
      self.covariance_matrixes = [1/N[k]*self.covariance_matrixes[k] for k in range(self.n_componets)]
      self.pi = [N[k]/len(X) for k in range(self.n_componets)]

  def likelyhood(self, extractedFeatures):
        '''
            The predicting function
                :param X: 2-d array numpy array
                    The data on which we must predict the clusters
        '''
        probas = []
        for n in range(len(extractedFeatures)):
            probas.append([self.multivariate_normal(extractedFeatures[n], self.mean_vector[k], self.covariance_matrixes[k])
                           for k in range(self.n_componets)])
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return np.max(cluster)

class DTW(ClassificatorInterface):
  def __init__(self):
    self.acquiredMFCC = np.array([])
    self.referenceMFCCMatrix = np.array([])

  def likelyhood(self, extractedFeatures):
    return 1

  def train(self, extractedFeaturesList):
    for mfccIndex in range(len(extractedFeaturesList)):
      logging.info(extractedFeaturesList[mfccIndex].shape)

# This classificator is used for testing program only
class FakeClassificator(ClassificatorInterface):
  def __init__(self):
    self.trained = False

  def likelyhood(self, extractedFeatures):
    return random.uniform(1, 1000)

  def train(self, X):
    return True


# EQ