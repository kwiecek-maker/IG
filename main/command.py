import os
import logging
import soundfile
import numpy as np
from unidecode import unidecode
from copy import copy


# Creates commands from the recording files. Associate command type, by name of the recording files
class CommandFactory:
    def __init__(self, path, classificator):
        self.path = path
        self.classificator = classificator
        self.commands = ['naprzod', 'odbierz', 'odrzuc', 'odslon', 'otworz', 'start', 'stop',
                         'wlacz', 'wstecz', 'wylacz', 'zamknij', 'zaslon', 'zaswiec', 'zgas']
        self.commandMap = dict((x, list()) for x in self.commands)  # map structure: {str(command) : [path1, path2, path3...]}
        self.rmsList = list()
        self.rmsNormalizeValue = 0

    def __repr__(self):
        outputString = ""
        for command in self.commands:
            outputString += str(command) +"\n"


    def rmsFromFile(self, filePath):
        data, samplerate = soundfile.read(filePath)
        data = data[:, 0].flatten()
        rmsCurrent = np.sqrt(np.mean(data ** 2))

        self.rmsList.append(rmsCurrent)

    # walks through files and categories and complete commandMap, and if command exist already, path is appended to list assigned to command
    # Clculate for each file rms value and append it to the list of rms values. Then take median of rms list,
    # and return it (this value will be needed to normalize signal in PreprocessUnit.normalize())
    def readCommands(self):
        for (directoryPath, diretoryName, fileName) in os.walk(self.path):
            for file in fileName:
                if file.endswith('.wav'):
                    fileDecode = unidecode(file.lower())  # decode from the polish characters + change to the lower letters
                    fileWithoutExtension = os.path.splitext(fileDecode)[0]

                    for key in self.commandMap.keys():
                        if fileWithoutExtension == key:
                            self.commandMap[key].append(os.path.join(directoryPath, file))
                            self.rmsFromFile(os.path.join(directoryPath, file))
                        else:
                            continue

        self.rmsNormalizeValue = np.median(self.rmsList)


    def createCommand(self, name, dataList):
        return Command(copy(self.classificator), name, dataList)

    # Returns List of all used commands in program
    def getCommandList(self):
        return self.commands

    # Returns rms value, which will be used in PreprocessUnit.normalize()
    def getRmsValue(self):
        return self.rmsNormalizeValue

# Manages all commands created by Command factory
class CommandManager:
    def __init__(self):
        self.commands = []

    def __repr__(self):
        output = "Comand Factory. Acquired commands: \n"
        for command in self.commands:
            output += str(command) +"\n"
        return output

    def __str__(self):
        return self.__repr__()

    # Acquires commands from Comands factory by passing command
    # List from command factory
    def acquireCommands(self, commandList):
        self.commands = commandList

    # Iterates over all self.commands, invoking likelyhood.
    # Finds command with the biggest likelyhood and returns it
    def recognize(self, extractedData):
        logging.warning("no implemented yet!")


# Mediator of the classificator
class Command:
    def __init__(self, classificator, commandName, dataList):
        self.classificator = classificator
        self.commandName = commandName
        self.dataList = dataList

    def __repr__(self):
        return "Command %s" % (self.commandName)

    def __str__(self):
        return self.__repr__()

    def train(self, dataList):
        self.classificator.train(dataList)

    def likelyhood(self, extractedFeatures):
        return self.classificator.likelyhood(extractedFeatures)

# EOF
