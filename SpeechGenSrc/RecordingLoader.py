import os
import soundfile as sf

# walks through directories and extract recordings as numpy array
class RecordingLoader:
    
    def __init__(self, relativePath=os.getcwd()):
        self.relativePath = relativePath
        self.recordingMap = {}
        self.recordingNames = []
    
    def __repr__(self):
        return "Recording Loader. Relative path: %s" % str(self.relativePath)

    def __str__(self):
        return self.__repr__()

    # walks trhough directories and create recordning map 
    # of recording paths for future extraction
    def acquireRecordingMap(self):
        recordingName = "defaultName"
        recordingPath = "defaultPAth"
        # Do the same for every found recording:
        self.recordingMap.update({recordingName : recordingPath})
        self.recordingNames.append(recordingName)

    def extractRecordings(self):
        outputRecordingMap = {}
        for name in self.recordingNames:
            data, sampleRate = sf.read(self.recordingMap[name])
            outputRecordingMap.update({name : data})
        return outputRecordingMap

    


