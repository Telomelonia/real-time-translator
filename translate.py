import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import threading
import queue
import time

class RealtimeTranslator:
    def __init__(self, model_id="Ivydata/whisper-small-japanese", language="ja"):
        """
        Initialize the real-time translator
        
        Args:
            model_id (str): Whisper model ID to use
            language (str): Language code (e.g., 'ja' for Japanese)
        """
        # Audio recording parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Volume detection parameters
        self.silence_threshold = 0.01  # Adjust this value based on your needs
        self.silence_duration = 2.0    # Seconds of silence before pausing
        self.last_sound_time = time.time()
        self.is_silent = False
        
        # Initialize Whisper model and processor
        print("Loading Whisper model...")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        
        # Configure model for specified language
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, 
            task="transcribe"
        )
        self.model.config.suppress_tokens = []
        print("Model loaded successfully!")

    def get_audio_level(self, audio_data):
        """
        Calculate the RMS amplitude of the audio data
        
        Args:
            audio_data (np.array): Audio data
        Returns:
            float: RMS amplitude
        """
        return np.sqrt(np.mean(np.square(audio_data)))

    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Status: {status}")
            
        # Calculate current audio level
        current_level = self.get_audio_level(indata)
        
        # Update silence state
        current_time = time.time()  # Use time module directly
        if current_level > self.silence_threshold:
            self.last_sound_time = current_time
            if self.is_silent:
                print("\nSound detected - resuming transcription...")
                self.is_silent = False
        elif not self.is_silent and (current_time - self.last_sound_time) > self.silence_duration:
            print("\nSilence detected - pausing transcription...")
            self.is_silent = True
        
        # Only queue audio data if not in silent state
        if not self.is_silent:
            self.audio_queue.put(indata.copy())

    def process_audio_chunk(self, audio_chunk):
        """
        Process a chunk of audio through Whisper model
        
        Args:
            audio_chunk (np.array): Audio data
        Returns:
            str: Transcribed text
        """
        # Prepare audio for model
        input_features = self.processor(
            audio_chunk, 
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features

        # Generate transcription
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription

    def start_recording(self, chunk_duration=5):
        """
        Start recording and transcribing audio in real-time
        
        Args:
            chunk_duration (int): Duration of each audio chunk in seconds
        """
        self.is_recording = True
        chunk_size = int(self.sample_rate * chunk_duration)
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self.audio_callback,
            blocksize=chunk_size
        ):
            print("Recording started! Speak now... (Press Ctrl+C to stop)")
            print(f"Silence threshold: {self.silence_threshold}")
            print(f"Silence duration before pause: {self.silence_duration} seconds")
            
            while self.is_recording:
                try:
                    # Get audio chunk from queue with timeout
                    try:
                        audio_chunk = self.audio_queue.get(timeout=1.0).flatten()
                    except queue.Empty:
                        continue
                    
                    # Process and transcribe
                    transcription = self.process_audio_chunk(audio_chunk)
                    
                    if transcription.strip():  # Only print non-empty transcriptions
                        print(f"Transcription: {transcription}")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error processing audio: {e}")
                    continue

    def stop_recording(self):
        """Stop the recording process"""
        self.is_recording = False
        print("\nRecording stopped!")

# Example usage
if __name__ == "__main__":
    # Initialize translator with Japanese model
    translator = RealtimeTranslator(
        model_id="Ivydata/whisper-small-japanese",
        language="ja"
    )
    
    try:
        # Start recording and transcribing in chunks of 5 seconds
        translator.start_recording(chunk_duration=5)
    except KeyboardInterrupt:
        translator.stop_recording()