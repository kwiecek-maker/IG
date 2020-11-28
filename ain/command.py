from copy import copy
class CommandFactory:
  def __init__(self, classificator):
    self.commandMap = {} # map structure:
                         # str(command) : [path1, path2, path3...]
    self.classificator = classificator
  # walks through files and categories and complete commandMap
  # if command exist already, path is appended to list assigned to command
  def readCommands(self):
    pass
  
  def makeCommand(self, name, dataList):
    return Command(copy(self.classificator), name, dataList)

   # creates List of used commands in program
  def makeCommandList(self):
    pass

# Mediator of the classificator
class Command:
  def __init__(self, classificator, commandName, dataList):
    self.classificator = classificator
    self.commandName = commandName
    self.dataList = dataList

  def train():
    pass

  def likelyhood(self, extractedFeatures):
    return self.classificator.likelyhood(extractedFeatures)
