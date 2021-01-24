import os
import logging
import soundfile
import numpy as np
from main.preprocessUnit import PreprocessUnit
from abc import ABC, abstractclassmethod
from main.featureExtractor import MFCC
from main.featureExtractor import ReferenceMFCC
from main.classificator import GMM
from unidecode import unidecode
import soundfile as sf
from copy import copy

global DEBUG
DEBUG = True

class CommandFactoryInterface(ABC):
    @abstractclassmethod
    def getCommandList(self):
        pass

    @abstractclassmethod
    def createCommand(self):
        pass

    @abstractclassmethod
    def readCommands(self):
        pass

class CommandReadingFactorGMM(CommandFactoryInterface):
    def __init__(self, classificator, trainingDataPath):
        self.classificator = classificator
        self.path = trainingDataPath
        self.commands = list()

    def readCommands(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line[:-2].split('\t')

                name = data[0]
                command = self.createCommand(name)

                means = self._arrayFromStr(data[2])
                variances = self._arrayFromStr(data[3])
                weights = self._arrayFromStr(data[4])
                command.acquireClassificatorParameters(means, variances, weights)
                command.trained = True
                self.commands.append(command)

    def createCommand(self, name):
        return Command(copy(self.classificator), name, list())

    def getCommandList(self, preprocessingUnit):
        return self.commands

    @staticmethod
    def _arrayFromStr(arrayStr):
        # getting rid of '[' and ']
        arrayStr = arrayStr[1:-1]

        array = arrayStr.split(',')
        return np.array([float(i) for i in array])

# Creates commands from the recording files. Associate
# command type, by name of the recording files
class CommandFactory(CommandFactoryInterface):
    def __init__(self, databasePath, classificator):
        self.path = databasePath
        self.classificator = classificator
        self.commandNames = []

        # map structure: {str(command) : [path1, path2, path3...]}
        self.commandMap = dict((x, list()) for x in self.commandNames)

        self.rmsList = list()
        self.rmsNormalizeTargetValue = 0

    def __repr__(self):
        outputString = "Command Factory, acquired commands: \n"
        for command in self.commands:
            outputString += str(command) +"\n"
        return outputString

    def __str__(self):
        return self.__repr__()

    def acquireRMSValue(self, filePath):
        data, samplerate = soundfile.read(filePath)
        data = self.flattenData(data)
        rmsCurrent = np.sqrt(np.mean(data ** 2))
        self.rmsList.append(rmsCurrent)

    def calculateGlobalRMSTarget(self):
        for key in self.commandMap.keys():
            for path in self.commandMap[key]:
                self.acquireRMSValue(path)
        self.rmsNormalizeTargetValue = np.median(self.rmsList)
        logging.info(" Calculated Global RMS Target: %f" % (self.rmsNormalizeTargetValue))

    # Walks through files and categories and complete associate
    # recording .wav files with command by the name of the recording file
    def readCommands(self):
        for (directoryPath, diretoryName, fileName) in os.walk(self.path):
            for file in fileName:
                if file.endswith('.wav'):

                    # decode from the polish characters + change to the lower letters
                    fileDecode = unidecode(file.lower())
                    fileWithoutExtension = os.path.splitext(fileDecode)[0]

                    path = os.getcwd() + "\\" + os.path.join(directoryPath, file)
                    if fileWithoutExtension in self.commandMap.keys():
                        for key in self.commandMap.keys():
                            if fileWithoutExtension == key:
                                self.commandMap[key].append(path)
                            else:
                                continue
                    else:
                        self.commandMap[fileWithoutExtension] = [path]
                        self.commandNames.append(fileWithoutExtension)

    def createCommand(self, name, dataList):
        return Command(copy(self.classificator), name, dataList)

    # Returns List of all used commandNames in program
    def getCommandList(self, preprocessUnit):
        outputCommands = list()
        for key in self.commandMap.keys():
            commandData = list()
            for path in self.commandMap[key]:
            # for path in self.commandMap[key][0:2]: #! This is for speed upgrade
                data, samplerate = sf.read(path)
                data = self.flattenData(data)

                preprocessUnit.samplingFrequency = samplerate
                preprocessedData = preprocessUnit.process(data)
                # downsamplingFrequency = preprocessUnit.downsamplingFrequency

                mfcc = ReferenceMFCC(preprocessedData, samplerate)
                # mfcc = MFCC(preprocessedData, samplerate=downsamplingFrequency, numberOfCepstras=12, numberOfMelFilters=24, numberOfFrequencyBins=512)
                mfccData = mfcc.extract().T

                commandData.append(mfccData)
            outputCommands.append(self.createCommand(key, commandData))
            message = " Command acquired: \"" + str(key) + "\" command"
            logging.info(message)
            if DEBUG:
                print(message)
        return outputCommands


    # Returns rms value, which will be used in PreprocessUnit.normalize()
    def getRmsValue(self):
        return self.rmsNormalizeTargetValue

    @staticmethod
    def flattenData(data):
        if len(data.shape) == 2:
            data = data[:, 0].flatten()
        else:
            data = data.flatten()
        return data

class CommandManagerInterface():
    @abstractclassmethod
    def acquireCommands(self, commandList):
        pass

    @abstractclassmethod
    def recognize(self, extractedData):
        pass

    @abstractclassmethod
    def train(self):
        pass

# Manages all commands created by Command factory
class CommandManager(CommandManagerInterface):
    def __init__(self, saveTrainedDataToCSV=True, trainingDataDestinationPath=None):
        self.commands = []
        self.saveTrainedDataToCSV = saveTrainedDataToCSV
        self.trainingDataDestinationPath = trainingDataDestinationPath

    def __repr__(self):
        output = "Comand Factory. Acquired commands: \n"
        for command in self.commandNames:
            output += str(command) +"\n"
        return output

    def __str__(self):
        return self.__repr__()

    # Acquires commands from Comands factory by passing command
    # List from command factory
    def acquireCommands(self, commandList):
        self.commands = commandList

    # Iterates over all self.commands, invoking likelihood.
    # Finds command with the biggest likelihood and returns it
    def recognize(self, extractedData):
        likelihood, index = 0, 0
        for commandIndex in range(len(self.commands)):
            templikelihood = self.commands[commandIndex].likelihood(extractedData)
            if  templikelihood > likelihood:
                likelihood = templikelihood
                index = commandIndex
        logging.info(" Recognized: \"%s\", with likelihood: %f" % (self.commands[index].name, likelihood))
        return self.commands[index].name

    def train(self):
        if self.saveTrainedDataToCSV: # Clear Training data
            open(self.trainingDataDestinationPath, 'w').close()

        for command in self.commands:

            if command.trained is False:
                message = " Training of \"" + str(command.name) + "\" command started!"
                logging.info(message)

                if DEBUG:
                    print(message)

                command.train()

                message = " Training of \"" + str(command.name) + "\" command ended!"
                logging.info(message)

                if DEBUG:
                    print(message)

                if self.saveTrainedDataToCSV:
                    with open(self.trainingDataDestinationPath, 'a') as f:
                        f.write(command.name + "\t" + str(command.classificator) + "\n")

            else:
                message = " Command: \"" + str(command.name) + "\" already trained"
                logging.info(message)


# Mediator of the classificator
class Command:
    def __init__(self, classificator, name, dataList, trained=False):
        self.classificator = classificator
        self.name = name
        self.dataList = dataList
        self.trained = trained

    def __repr__(self):
        return "Command %s" % (self.commandName)

    def __str__(self):
        return self.__repr__()

    def train(self):
        self.classificator.train(self.dataList)

    def likelihood(self, extractedFeatures):
        return self.classificator.likelihood(extractedFeatures)

    def acquireClassificatorParameters(self, means, variances, weights):

        if len(means) != len(variances) or len(variances) != len(weights):
            raise AttributeError("length of each parameters must be equal")

        self.classificator.means = means
        self.classificator.variances = variances
        self.classificator.weights = weights

# EOF
