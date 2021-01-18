import main.command as command
from main.classificator import FakeClassificator
import logging


# Clearing the logs
open('logging.log', 'w').close()

logging.basicConfig(filename = 'logging.log', level = logging.DEBUG)
logging.info(" Starting Command Test")

classificator = FakeClassificator()

commandFactory = command.CommandFactory('database', classificator)
commandFactory.readCommands()
commandFactory.calculateGlobalRMSTarget()
commandList = commandFactory.getCommandList()

