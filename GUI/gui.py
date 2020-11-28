from abc import ABC, abstractmethod
import logging

class GuiInterface(ABC):
  @abstractmethod
  def handle(self):
    pass
  @abstractmethod
  def checkEvents(self):
    pass

# Runs animation based on the given command 
class GUISmartHome(GuiInterface):
  def __init__(self):
    self.GUIQueue = None

  def handle(self, command):
    pass
  # checks windows events
  def checkEvents(self):
    pass

  def isCommandAvailable(self):
    pass
  
  # TODO: establish all utilities that gui must do with team
