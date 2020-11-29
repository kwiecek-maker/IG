from copy import copy
import logging

# Creates commands from the recording files. Associate command type 
# by name of the recording files 
class CommandFactory:
  def __init__(self, classificator):
    self.commandMap = {} # map structure:
                         # str(command) : [path1, path2, path3...]
    self.classificator = classificator
  # walks through files and categories and complete commandMap
  # if command exist already, path is appended to list assigned to command
  def readCommands(self):
    pass
  
  def createCommand(self, name, dataList):
    return Command(copy(self.classificator), name, dataList)

   # returns List of all used commands in program
  def getCommandList(self):
    pass

# Manages all commands created by Command factory 
class CommandManager:
  def __init__(self):
    self.commands = []
  
  # acquires commands from Comands factory by passing command
  # list from command factory 
  def acquireCommands(self, commandList):
    self.commands = commandList

  # iterates over all self.commands, invoking likelyhood.
  # finds command with the biggest likelyhood and returns it
  def recognize(self, extractedData):
    logging.warning("no implemented yet!")

# Mediator of the classificator. 
class Command:
  def __init__(self, classificator, commandName, dataList):
    self.classificator = classificator
    self.commandName = commandName
    self.dataList = dataList

  def train(self, dataList):
    self.classificator.train(dataList)
    
  def likelyhood(self, extractedFeatures):
    return self.classificator.likelyhood(extractedFeatures)

# EOF