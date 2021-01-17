from abc import ABC, abstractmethod
import logging
import queue

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
    self.GUIQueue = queue.Queue()

  def handle(self, command):
    pass
  # checks windows events
  def checkEvents(self):
    pass

  def isCommandAvailable(self):
    pass

  def putIntoQueue(self, command):
    self.GUIQueue.put(command)

  # TODO: establish all utilities that gui must do with team
