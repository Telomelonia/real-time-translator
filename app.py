# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import multiprocessing
import threading
import os

# Import the services directly
from transcribe import AudioTranscriber  # Remove services. prefix
from translate import Translator        # Remove services. prefix

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Added secret key for Flask
socketio = SocketIO(app, cors_allowed_origins="*")

# Global queues for communication
transcription_queue = multiprocessing.Queue()
broadcast_queue = multiprocessing.Queue()

# Global variables for cleanup
transcriber = None
should_run = True

def transcription_to_broadcast():
    """Thread to move transcriptions to broadcast queue"""
    global should_run
    while should_run:
        try:
            item = transcription_queue.get(timeout=1)  # Add timeout for checking should_run
            if item is None:
                break
            broadcast_queue.put(item)
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            print(f"Queue transfer error: {e}")
            continue

def broadcast_translations():
    """Thread to broadcast translations to all clients"""
    global should_run
    translator = Translator()
    print("Translation service initialized")
    
    while should_run:
        try:
            item = broadcast_queue.get(timeout=1)  # Add timeout for checking should_run
            if item is None:
                break
            
            translation = translator.translate(item['text'])
            print(f"\nOriginal: {item['text']}")
            print(f"Translation: {translation}")
            
            socketio.emit('translation', {
                'timestamp': item['timestamp'],
                'original': item['text'],
                'translation': translation
            })
            
        except multiprocessing.queues.Empty:
            continue
        except Exception as e:
            print(f"Broadcasting error: {e}")
            continue

@app.route('/')
def index():
    return render_template('index.html')

def start_transcriber():
    """Start the transcription process"""
    global transcriber, should_run
    transcriber = AudioTranscriber(transcription_queue)
    try:
        transcriber.start_recording()
        print("Transcriber started successfully")
        while should_run:  # Keep thread alive
            threading.Event().wait(1)
    except Exception as e:
        print(f"Transcriber error: {e}")
    finally:
        if transcriber:
            transcriber.stop_recording()

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def cleanup():
    """Cleanup function to stop all processes and threads"""
    global should_run, transcriber
    print("\nCleaning up...")
    should_run = False
    
    # Stop transcriber
    if transcriber:
        transcriber.stop_recording()
    
    # Signal threads to stop
    transcription_queue.put(None)
    broadcast_queue.put(None)

if __name__ == '__main__':
    try:
        # Start the transcription process
        transcriber_thread = threading.Thread(target=start_transcriber)
        transcriber_thread.daemon = True
        transcriber_thread.start()
        
        # Start the queue transfer thread
        transfer_thread = threading.Thread(target=transcription_to_broadcast)
        transfer_thread.daemon = True
        transfer_thread.start()
        
        # Start the broadcast thread
        broadcast_thread = threading.Thread(target=broadcast_translations)
        broadcast_thread.daemon = True
        broadcast_thread.start()
        
        print("All services started successfully")
        print("Access the application at http://localhost:5000")
        
        # Run the Flask app
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Startup error: {e}")
    finally:
        cleanup()