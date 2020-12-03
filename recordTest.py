from keyboard import record
import main.recorder as recorder

import logging
import keyboard
import threading


open('record_test.log', 'w').close()

logging.basicConfig(filename = 'record_test.log', level = logging.DEBUG)
logging.info("Starting record_test")
rec = recorder.Recorder(0.05)

def plotRecordingThread():
  t = threading.current_thread()
  logging.info("Started plotting recording %s", t.getName())
  while True:
    if rec.isDataAvailable():
      logging.info("EXPORTED: " + str(rec.exportRecording()))
    if keyboard.is_pressed('q'):
      break
  logging.info("Ending recording %s", t.getName())

def recordingThread():
  t = threading.current_thread()
  logging.info("Starting recording %s", t.getName())
  rec.run()
  logging.info("Ending recording %s", t.getName())

def main():
  threads = []
  threads.append(threading.Thread(target=recordingThread))
  threads.append(threading.Thread(target=plotRecordingThread))
  for thread in threads:
    thread.start()
    
  for thread in threads:
    thread.join()
    
if __name__ == "__main__":
  main()


