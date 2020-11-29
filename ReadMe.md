# Speech recognition program made by students from AGH. 

Program listen for basic commands via microphone, collects recording thats exceeds given rms level,
Extract Features from collected recording and assigns acquired data to the command that has the biggest likelihood. 

## How program works:
- Phase 1: Acquiring Data

  1. Command Factory Walks through directories and collects paths of audio files for training.
  It assigns audio files with the same name to command

  2. From acquired data list it creates command List and pass it to Command Manager

- Phase 2: Training Models

  1. Iterates through commands in commandManager and performs training of all data.
    TODO: we need to check which of the settings of trained data suits the most our program

- Phase 3: Starting program 

  1. Initialize Gui thread
  2. Initialize dataCalculation thread
  3. initialize recording Thread

  4. Run all Threads 
