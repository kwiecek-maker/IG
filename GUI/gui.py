from abc import ABC, abstractclassmethod

class GuiInterface(ABC):
  @abstractclassmethod
  def handle(self):
    pass
  @abstractclassmethod
  def checkEvens(self):
    pass

# Runs animation based on the given command 
class GUISmartHome(GuiInterface):
  def __init__(self):
    self.GUIQueue = None
    # TODO: how to do commands handling 
    def handle(self, command):
      pass
    # checks windows events
    def checkEvents(self):
      pass
    # TODO: establish all utilities that gui must do with team
