import numpy as np
import pyaudio
import threading
import queue
import torch
from faster_whisper import WhisperModel
import time

class AudioTranscriber:
    def __init__(self):
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )
        
        # Audio recording parameters
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 0.005
        
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def start_recording(self):
        self.is_recording = True
        
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording started (Press Ctrl+C to stop)")
        print("Speak in Japanese...")
        
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
        self.transcription_thread = threading.Thread(target=self._transcribe_audio)
        self.transcription_thread.start()
    
    def stop_recording(self):
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("\nRecording stopped")
    
    def _record_audio(self):
        while self.is_recording:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                if np.abs(audio_chunk).mean() > self.SILENCE_THRESHOLD:
                    self.audio_queue.put(audio_chunk)
                    
            except Exception as e:
                print(f"Error recording audio: {e}")
                continue
    
    def _transcribe_audio(self):
        audio_buffer = []
        last_transcription_time = time.time()
        silence_time = 0
        
        while self.is_recording:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.5)
                audio_buffer.append(audio_chunk)
                silence_time = 0
                
                buffer_duration = len(audio_buffer) * self.CHUNK / self.RATE
                
                if buffer_duration >= 2.0:
                    self._process_buffer(audio_buffer)
                    audio_buffer = []
                    last_transcription_time = time.time()
                    
            except queue.Empty:
                silence_time += 0.5
                if silence_time >= 1.0 and audio_buffer:
                    self._process_buffer(audio_buffer)
                    audio_buffer = []
                    last_transcription_time = time.time()
                continue
    
    def _process_buffer(self, audio_buffer):
        try:
            audio_data = np.concatenate(audio_buffer)
            segments, _ = self.model.transcribe(
                audio_data,
                language="ja",
                beam_size=5
            )
            
            for segment in segments:
                print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")

def main():
    transcriber = AudioTranscriber()
    
    try:
        transcriber.start_recording()
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        transcriber.stop_recording()
        print("Transcription ended")

if __name__ == "__main__":
    main()