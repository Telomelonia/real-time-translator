import pyaudio
import wave
from pydub import AudioSegment
import os

def test_microphone(duration=5, output_filename="test_recording"):
    # Audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Print available input devices
    print("\nAvailable Input Devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {dev_info['name']}")
    
    # Open audio stream
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    print(f"\nRecording for {duration} seconds...")
    frames = []
    
    # Record audio
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        # Print audio level every second
        if i % int(RATE / CHUNK) == 0:
            print(f"Recording... {i//(RATE/CHUNK)}s")
    
    print("\nFinished recording!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save as WAV first
    wav_filename = f"{output_filename}.wav"
    with wave.open(wav_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    # Convert to MP3
    mp3_filename = f"{output_filename}.mp3"
    audio_segment = AudioSegment.from_wav(wav_filename)
    audio_segment.export(mp3_filename, format="mp3")
    
    # Remove WAV file
    os.remove(wav_filename)
    
    print(f"\nRecording saved as {mp3_filename}")
    print(f"File size: {os.path.getsize(mp3_filename) / 1024:.2f} KB")
    
    return mp3_filename

if __name__ == "__main__":
    try:
        # Record audio
        output_file = test_microphone(duration=5, output_filename="microphone_test")
        print(f"\nTest completed successfully! Check {output_file}")
        
    except Exception as e:
        print(f"\nError during recording: {e}")
        print("Please make sure your microphone is properly connected and accessible.")