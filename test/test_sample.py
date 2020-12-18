import soundfile

class TestClass:
    def test_baby(self):
        assert True == True
        
def test_openRecoring():
    data, samplerate = soundfile.read("testRecordings/testRecording.wav")
