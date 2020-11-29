# Speech recognition program 
## made by students from AGH ;-)

Program listen for basic commands via microphone, collects recording thats exceeds given RMS level,
Extract Features from collected recording and assigns acquired data to right command.

# Control

Q - quits program

## How program works:
- Phase 1: Acquiring Data

  1. Command Factory Walks through directories and collects paths of audio files for future training Phase.
  It assigns audio files with the same name to one command.

  2. From acquired data list it creates command List and pass it to Command Manager

- Phase 2: Training Models
  
  1. With usage of preprocessUnit it prepares all data to be trained

  2. Extract all features from given  data and assign them to commands

  3. Iterates through commands in commandManager and performs training of all data.

- Phase 3: Starting program 

  1. Runs Gui Thread:
    Gui handles commands with instruction obtained from given commands dictionary.
    
  2. Runs Data calculation Thread
    Performs recognition of the recording.

  3. Runs Recording Thread
    Acquires recording buffers that exceeds given RMS level

