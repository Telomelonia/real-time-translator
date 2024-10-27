import pyaudio
import wave
import whisper
import numpy as np
import threading
import time
import os

# Load Whisper model globally
model = whisper.load_model("base")

def list_audio_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    
    for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print(f"Input Device id {i} - {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
    
    p.terminate()

class AudioRecorder:
    def __init__(self, filename="temp.wav", sample_rate=16000, channels=1, chunk=1024, device_index=None):
        self.filename = filename
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=self.channels,
                                      rate=self.sample_rate,
                                      input=True,
                                      input_device_index=self.device_index,
                                      frames_per_buffer=self.chunk)
        self.is_recording = True
        self.frames = []
        print("Recording started...")

    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        print("Recording stopped.")

        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))

    def record_chunk(self):
        if self.is_recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

def transcribe_and_translate(audio_file):
    result = model.transcribe(audio_file, task="translate", language="ja")
    return result["text"]

def real_time_translation(device_index):
    recorder = AudioRecorder(device_index=device_index)
    recorder.start_recording()

    print("Real-time translation started. Press Enter to stop.")
    
    def record_loop():
        while recorder.is_recording:
            recorder.record_chunk()
            time.sleep(0.1)  # Small delay to reduce CPU usage

    record_thread = threading.Thread(target=record_loop)
    record_thread.start()

    while True:
        if input() == '':
            break
        recorder.stop_recording()
        if os.path.exists(recorder.filename) and os.path.getsize(recorder.filename) > 0:
            translated_text = transcribe_and_translate(recorder.filename)
            print("Translated text:", translated_text)
        recorder.start_recording()

    recorder.stop_recording()
    record_thread.join()

# Example usage
list_audio_devices()
selected_device = int(input("Enter the device ID you want to use for recording: "))
real_time_translation(selected_device)