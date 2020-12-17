from abc import ABC


class ClassificatorInterface(ABC):
    def __init__(self, mfccFeaturesArray):
        self.mfccFeaturesArray = mfccFeaturesArray
        self.numberOfFrames = mfccFeaturesArray.size[1]
        self.numberOfCepstras = mfccFeaturesArray.size[0]

    # Returns likely hood how well function
    # inside classificator approximates histogram of given data
    # @abstractclassmethod
    # def likelyhood(self, extractedFeatures):
    #   pass
    #
    # # creates function that approximates histogram of given data List
    # @abstractclassmethod
    # def train(self, extractedFeaturesList):
    pass

# Dynamic Time Warping algorithm
class DTW(ClassificatorInterface):
    def __init__(self, mfccFeaturesArray):
        super(DTW, self).__init__(mfccFeaturesArray)
        pass

    def mfcc_preprocess(self):
        for frame_num in range(self.numberOfFrames):
            pass

    # def likelyhood(self, extractedFeatures):
    #   pass
    #
    # def train(self, extractedFeaturesList):
    #   pass

# GMM creates gaussian mixture that approximates given data
# |nComponents| - The number of mixture components.
# |tolerance| - The convergence threshold. EM iterations will stop when the
# lower bound average gain is below this threshold.
# |maxIteration| - The number of EM iterations to perform.
# class GMM(ClassificatorInterface):
#   def __init__(self, nComponents=1, tolerance=1e-3, maxIterations=100 ):
#     self.nComponents = nComponents
#     self.tolerance = tolerance
#     self.maxIterations = maxIterations
#
#   def likelyhood(self, extractedFeatures):
#     pass
#
#   def train(self, extractedFeaturesList):
#     pass

# EOF