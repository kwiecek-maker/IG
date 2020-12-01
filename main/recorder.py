import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import queue
import sounddevice as sd
import sys

from matplotlib.animation import FuncAnimation

# Helper function for argument parsing.
def intOrString(text):
    try:
        return int(text)
    except ValueError:
        return text

# Recorder is listening for commands, starts/stop aquairing them
# and exports recorded data as numpy array
class Recorder:

  def __init__(self, thresholdLevel):
    self.threshold = thresholdLevel
    
    self.setUpArgParsingForAudioDevice()
    
    self.bufferQueue = queue.Queue()
    self.acquiredBuffersQueue = queue.Queue()
    self.acquiredRecordingQueue = queue.Queue() 
  
  def exportRecording(self):
    try:
      return self.acquiredRecordingQueue.get_nowait()
    except queue.Empty:
      return None
  
  def isDataAvailable(self):
    return self.acquiredRecordingQueue.empty()
  
  def run(self):
    try:
      self.checkSampleRate()

      bufferLength = int(self.args.window * self.args.samplerate / (1000 * self.args.downsample))
      
      self.plotData = np.zeros((bufferLength, len(self.args.channels)))
      fig = self.prepareFancyPlot(self.plotData)

      recordingStream = sd.InputStream(
          device=self.args.device, channels=max(self.args.channels),
          samplerate=self.args.samplerate, callback=self.collectAudioBuffer)
      
      animation = FuncAnimation(self.fig, self.updateRecorder, interval=self.args.interval, blit=True)
      with recordingStream:
          plt.show()
    except Exception as e:
        self.childParser.exit(type(e).__name__ + ': ' + str(e))
  
  def checkSampleRate(self):
    if self.args.samplerate is None:
          device_info = sd.query_devices(self.args.device, 'input')
          self.args.samplerate = device_info['default_samplerate']
  
  def setUpArgParsingForAudioDevice(self):
    [self.args, self.remaining, self.parentParser] = self.initParentParser()
    self.childParser = self.initChildParser(self.parentParser)
    self.prepareChildParser(self.childParser)
    self.args = self.childParser.parse_args(self.remaining)
    
    # Channel mapping for audio recording
    if any(channel < 1 for channel in self.args.channels):
        self.parser.error('|channel|: must be >= 1')
    self.channelMapping = [channel - 1 for channel in self.args.channels]  
    
    
  def collectAudioBuffer(self,inData, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a necessary copy.
    self.bufferQueue.put(inData[::self.args.downsample, self.channelMapping])
  
  def prepareFancyPlot(self, plotData):
    self.fig, self.ax = plt.subplots()
    self.lines = self.ax.plot(plotData)
    if len(self.args.channels) > 1:
        self.ax.legend(['channel {}'.format(c) for c in self.args.channels],
                  loc='lower left', ncol=len(self.args.channels))
    self.ax.axis((0, len(plotData), -1, 1))
    self.ax.set_yticks([0])
    self.ax.yaxis.grid(True)
    self.ax.tick_params(bottom=False, top=False, labelbottom=False,
                    right=False, left=False, labelleft=False)
    self.fig.tight_layout(pad=0)
    return [self.fig, self.ax, self.lines]
  
  def updateRecorder(self, frame):
    while True:
        try:
            data = self.bufferQueue.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        self.plotData = np.roll(self.plotData, -shift, axis=0)
        self.plotData[-shift:, :] = data
    for column, line in enumerate(self.lines):
        line.set_ydata(self.plotData[:, column])
    return self.lines
  
  # check if given audio buffer exceeds threshold 
  @staticmethod
  def isBufferLevelAboveThreshold(buffer):
    pass
  
  # Creates argument parser for all audio devices available.
  @staticmethod
  def initParentParser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
      logging.info((sd.query_devices()))
      parser.exit(0)
    return [args, remaining, parser]
  
  # Inits parser for set up audio device.
  @staticmethod
  def initChildParser(parentParser):
    parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parentParser])
    return parser
  
  # Adds all required args used for audio set up.
  @staticmethod
  def prepareChildParser(parser):
    parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
    parser.add_argument(
    '-d', '--device', type=intOrString,
    help='input device (numeric ID or substring)')
    parser.add_argument(
    '-w', '--window', type=float, default=256, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
    parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
    parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
    parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
  #EOF