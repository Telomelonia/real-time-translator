import sys
print("Python path:", sys.executable)
print("Python version:", sys.version)

try:
    import whisper
    print("Whisper version:", whisper.__version__)
    model = whisper.load_model("tiny")  # Try loading the smallest model first
    print("Successfully loaded whisper model!")
except Exception as e:
    print("Error:", str(e))
