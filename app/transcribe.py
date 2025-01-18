import numpy as np
import pyaudio
import threading
import queue
import torch
from faster_whisper import WhisperModel
import time

class AudioTranscriber:
    def __init__(self):
        # Initialize Faster Whisper model
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )
        
        # Audio recording parameters
        self.CHUNK = 1024 * 2  # Increased chunk size
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.SILENCE_THRESHOLD = 0.005  # Adjust this value based on your microphone
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Create a queue for audio chunks
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.is_recording = True
        
        # Open stream
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )
        
        print("* Recording started (Press Ctrl+C to stop)")
        print("* Speak in Japanese...")
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.start()
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self._transcribe_audio)
        self.transcription_thread.start()
    
    def stop_recording(self):
        """Stop recording audio"""
        self.is_recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("\n* Recording stopped")
    
    def _record_audio(self):
        """Record audio chunks and put them in queue"""
        while self.is_recording:
            try:
                # Read chunk from stream
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Simple noise gate
                print("Audio level:", np.abs(audio_chunk).mean())
                if np.abs(audio_chunk).mean() > self.SILENCE_THRESHOLD:
                    self.audio_queue.put(audio_chunk)
                    print("Queue size:", self.audio_queue.qsize())
                    
            except Exception as e:
                print(f"Error recording audio: {e}")
                continue
    
    def _transcribe_audio(self):
        """Transcribe audio chunks from queue"""
        audio_buffer = []
        last_transcription_time = time.time()
        silence_time = 0
        
        while self.is_recording:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=0.5)
                audio_buffer.append(audio_chunk)
                silence_time = 0
                
                # Calculate buffer duration
                buffer_duration = len(audio_buffer) * self.CHUNK / self.RATE
                
                # Transcribe if buffer is large enough (2 seconds) or if enough time has passed
                if buffer_duration >= 2.0:
                    self._process_buffer(audio_buffer)
                    audio_buffer = []
                    last_transcription_time = time.time()
                    
            except queue.Empty:
                silence_time += 0.5
                # Process remaining audio after 1 second of silence
                if silence_time >= 1.0 and audio_buffer:
                    self._process_buffer(audio_buffer)
                    audio_buffer = []
                    last_transcription_time = time.time()
                continue
    
    def _process_buffer(self, audio_buffer):
        """Process and transcribe the audio buffer"""
        try:
            # Concatenate audio chunks
            audio_data = np.concatenate(audio_buffer)
            
            # Transcribe using Faster Whisper
            segments, _ = self.model.transcribe(
                audio_data,
                language="ja",
                beam_size=5
            )
            
            # Print transcribed text
            for segment in segments:
                print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")

def main():
    # Create transcriber instance
    transcriber = AudioTranscriber()
    
    try:
        # Start recording and transcription
        transcriber.start_recording()
        
        # Keep running until user interrupts
        while True:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        # Stop recording on keyboard interrupt
        transcriber.stop_recording()
        print("\nTranscription ended")

if __name__ == "__main__":
    main()