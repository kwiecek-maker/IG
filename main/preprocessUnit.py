# Prepares data for futher calculations
class PreprocessUnit:
  def __init__(self, desiredLoudnessLevel, downsamplingFrequency):
    pass
  # deletes constant component from the recording 
  def deleteAverage(self, inputArrayRecording):
    pass
  # normalize given recording to the self.desiredLoudnessLevel
  def normalize(self, inputArrayRecording):
    pass
  # downsample recording to the self.downsamlingFrequency
  def downsample(self, inputArrayRecording):
    pass
  # performs all required processing using methods in this class
  def process(self, inputArrayRecording):
    pass
