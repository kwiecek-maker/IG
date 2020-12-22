"""
this file should be run with addictional command arugment. For example:
$ python decode.py "command"

It will automaticly play decoded signal.
"""

import sys

# change given name to the associated LPC coefficient file and LPC coefficient error
def transposeName(name):
    return ["", ""]

# walks trhough directoires and dread files associated with 
# given filename. Returns None when files are not found.
def readTxtfiles(filename):
    return filename 

# decode txt files from database files
class Decoder:

    def __init__(self, LPCmatrix, LPCerror):
        self.LPCmatrix = LPCMatrix
        self.LPCerror = LPCerror
    
    def generateRecording(self):
        return False
    
    def playRecording(self):
        return False

if __name__ == "__main__":
    name = sys.argv[1]
    [LPCmatrixName, LPCerrrorName] = transposeName(name)

    LPCMatrix = readTxtfiles(LPCmatrixName)
    LPCError = readTxtfiles(LPCerrrorName)

    decoder = Decoder(LPCMatrix, LPCError)
    decoder.generateRecording()
    decoder.playRecording()