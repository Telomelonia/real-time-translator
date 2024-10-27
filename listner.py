import sounddevice as sd
import numpy as np
from pydub import AudioSegment
import io
import wave
import tempfile
import os
from datetime import datetime

class AudioRecorder:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = None
        
    def record(self, duration_seconds):
        """
        Record audio from microphone for specified duration
        
        Args:
            duration_seconds (float): Recording duration in seconds
        """
        print("Recording started...")
        
        # Capture audio using sounddevice
        recording = sd.rec(
            int(duration_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16
        )
        
        # Wait until recording is complete
        sd.wait()
        self.recording = recording
        print("Recording finished!")
        
    def save_mp3(self, output_path=None):
        """
        Save the recorded audio as MP3
        
        Args:
            output_path (str): Path to save MP3 file. If None, generates timestamp-based filename
        """
        if self.recording is None:
            raise ValueError("No recording available - please record audio first")
            
        if output_path is None:
            # Generate filename with timestamp if none provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recording_{timestamp}.mp3"
        
        # First save as WAV using temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(self.recording.tobytes())
        
        # Convert WAV to MP3 using pydub
        audio = AudioSegment.from_wav(temp_wav.name)
        audio.export(output_path, format="mp3")
        
        # Clean up temporary WAV file
        os.unlink(temp_wav.name)
        print(f"Audio saved to: {output_path}")
        
    def play(self):
        """Play back the recorded audio"""
        if self.recording is None:
            raise ValueError("No recording available - please record audio first")
            
        print("Playing recording...")
        sd.play(self.recording, self.sample_rate)
        sd.wait()
        print("Playback finished!")

# Example usage:
if __name__ == "__main__":
    recorder = AudioRecorder()
    
    # Record 5 seconds of audio
    recorder.record(5)
    
    # Play it back
    recorder.play()
    
    # Save as MP3
    recorder.save_mp3("my_recording.mp3")