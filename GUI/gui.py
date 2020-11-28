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
  # TODO: how to do commands handling 
  def handle(self, command):
    logging.warning("handle not implemented!")
  # checks windows events
  def checkEvents(self):
    logging.warning("checkEvents not implemented!")
  # TODO: establish all utilities that gui must do with team
