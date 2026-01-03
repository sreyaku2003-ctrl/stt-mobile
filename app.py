from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
import os
import tempfile
from pydub import AudioSegment
from pydub.utils import which
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

# Set ffmpeg path (needed for audio conversion)
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Azure Speech Service credentials
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_REGION = os.getenv('AZURE_REGION', 'centralindia')

# Thread pool for handling multiple requests
executor = ThreadPoolExecutor(max_workers=10)


def convert_to_wav(audio_file, filename, request_id):
    """Convert any audio format to WAV (16kHz, 16-bit, mono) for Azure"""
    try:
        print(f"[{request_id}] 🔄 Converting audio: {filename}")
        
        # Create unique temp file for this request
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1], prefix=f'input_{request_id}_')
        audio_file.save(temp_input.name)
        temp_input_path = temp_input.name
        temp_input.close()
        
        # Load audio with pydub (supports many formats)
        audio = AudioSegment.from_file(temp_input_path)
        
        # Convert to Azure-compatible format: 16kHz, 16-bit, mono
        audio = audio.set_frame_rate(16000)
        audio = audio.set_sample_width(2)  # 16-bit
        audio = audio.set_channels(1)  # mono
        
        # Save as WAV with unique name
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix=f'output_{request_id}_')
        audio.export(temp_output.name, format='wav')
        temp_output.close()
        
        # Clean up input file
        os.unlink(temp_input_path)
        
        duration = len(audio) / 1000.0  # Duration in seconds
        print(f"[{request_id}] ✅ Converted to WAV: 16kHz, 16-bit, mono | Duration: {duration:.2f}s")
        return temp_output.name, duration
        
    except Exception as e:
        print(f"[{request_id}] ❌ Conversion error: {str(e)}")
        raise


def safe_delete_file(filename, request_id):
    """Safely delete a file with retry logic for Windows file locks"""
    if filename and os.path.exists(filename):
        for attempt in range(3):
            try:
                os.unlink(filename)
                print(f"[{request_id}] 🗑️ Deleted temp file")
                return True
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)
                else:
                    print(f"[{request_id}] ⚠️ Could not delete temp file: {filename}")
                    return False
    return True


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Azure Speech-to-Text API',
        'version': '1.0',
        'provider': 'Azure Cognitive Services',
        'default_language': 'en-IN',
        'concurrent_support': True
    }), 200


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Speech-to-Text endpoint using Azure - supports concurrent requests"""
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    temp_filename = None
    
    try:
        if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "your_azure_key_here":
            return jsonify({'error': 'Azure Speech API key not configured'}), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Default language is now en-IN (Indian English)
        language = request.form.get('language', 'en-IN')

        print(f"\n[{request_id}] {'='*50}")
        print(f"[{request_id}] 📝 Transcription Request:")
        print(f"[{request_id}]    File: {audio_file.filename}")
        print(f"[{request_id}]    Language: {language}")
        print(f"[{request_id}]    Provider: Azure Speech Services")
        print(f"[{request_id}] {'='*50}\n")

        # Convert audio to Azure-compatible WAV format
        try:
            temp_filename, duration = convert_to_wav(audio_file, audio_file.filename, request_id)
            print(f"[{request_id}] 🎵 Audio duration: {duration:.2f} seconds")
        except Exception as conv_error:
            return jsonify({
                'success': False,
                'error': f'Audio format conversion failed: {str(conv_error)}',
                'message': 'Please upload a valid audio file (WAV, MP3, OGG, M4A, etc.)',
                'request_id': request_id
            }), 400

        # Configure Azure Speech Service
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        speech_config.speech_recognition_language = language

        # Create audio configuration from converted WAV file
        audio_config = speechsdk.AudioConfig(filename=temp_filename)

        # Use continuous recognition to get complete transcription
        print(f"[{request_id}] 🔄 Using continuous recognition ({duration:.2f}s)")
        
        # Create speech recognizer
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        all_results = []
        done = False
        error_occurred = False
        error_details = None

        def handle_recognizing(evt):
            """Callback for partial results"""
            print(f"[{request_id}] 🔄 Recognizing: {evt.result.text}")

        def handle_final_result(evt):
            """Callback for recognized speech"""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                all_results.append(evt.result.text)
                print(f"[{request_id}] 📝 Recognized: {evt.result.text}")
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                print(f"[{request_id}] ⚠️ No match: {evt.result.no_match_details}")

        def handle_canceled(evt):
            """Callback for cancellation"""
            nonlocal done, error_occurred, error_details
            print(f"[{request_id}] ⚠️ Recognition canceled: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                error_occurred = True
                error_details = evt.error_details
                print(f"[{request_id}] ❌ Error details: {evt.error_details}")
            done = True

        def stop_continuous(evt):
            """Callback to stop continuous recognition"""
            nonlocal done
            done = True
            print(f"[{request_id}] ✅ Recognition completed")

        # Connect callbacks
        speech_recognizer.recognizing.connect(handle_recognizing)
        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.session_stopped.connect(stop_continuous)
        speech_recognizer.canceled.connect(handle_canceled)

        # Start continuous recognition
        speech_recognizer.start_continuous_recognition()

        # Wait until recognition is done with timeout
        timeout = max(duration + 10, 60)  # At least 60 seconds or duration + 10 seconds
        start_time = time.time()
        
        while not done:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                print(f"[{request_id}] ⚠️ Timeout reached after {timeout}s")
                break

        # Stop recognition
        speech_recognizer.stop_continuous_recognition()
        
        # Check for errors
        if error_occurred:
            speech_recognizer = None
            audio_config = None
            safe_delete_file(temp_filename, request_id)
            return jsonify({
                'success': False,
                'error': f'Recognition error: {error_details}',
                'message': 'Speech recognition failed',
                'request_id': request_id
            }), 500

        # Close the recognizer to release file handles
        speech_recognizer = None
        audio_config = None

        # Clean up temp file
        safe_delete_file(temp_filename, request_id)

        if all_results:
            full_text = " ".join(all_results)
            print(f"[{request_id}] ✅ Full transcription: {full_text}\n")
            
            return jsonify({
                'success': True,
                'text': full_text,
                'language': language,
                'duration': duration,
                'provider': 'Azure Speech Services',
                'method': 'continuous_recognition',
                'request_id': request_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'No speech could be recognized',
                'message': 'Please ensure the audio contains clear speech',
                'request_id': request_id
            }), 400

    except Exception as e:
        # Clean up temp file in case of error
        safe_delete_file(temp_filename, request_id)
        
        print(f"[{request_id}] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to transcribe audio',
            'request_id': request_id
        }), 500


@app.route('/transcribe-with-timestamps', methods=['POST'])
def transcribe_with_timestamps():
    """Speech-to-Text with detailed results including timestamps"""
    request_id = str(uuid.uuid4())[:8]
    temp_filename = None
    
    try:
        if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "your_azure_key_here":
            return jsonify({'error': 'Azure Speech API key not configured'}), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        language = request.form.get('language', 'en-IN')

        # Convert audio to Azure-compatible WAV format
        try:
            temp_filename, duration = convert_to_wav(audio_file, audio_file.filename, request_id)
        except Exception as conv_error:
            return jsonify({
                'success': False,
                'error': f'Audio format conversion failed: {str(conv_error)}',
                'message': 'Please upload a valid audio file',
                'request_id': request_id
            }), 400

        # Configure Azure Speech Service
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        speech_config.speech_recognition_language = language
        speech_config.request_word_level_timestamps()
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        # Create audio configuration
        audio_config = speechsdk.AudioConfig(filename=temp_filename)

        # Always use continuous recognition to get complete transcription
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        all_results = []
        all_segments = []
        done = False
        error_occurred = False
        error_details = None

        def handle_final_result(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                import json
                detailed = json.loads(evt.result.json)
                all_results.append(evt.result.text)
                if 'NBest' in detailed and detailed['NBest']:
                    all_segments.append(detailed['NBest'][0])

        def handle_canceled(evt):
            nonlocal done, error_occurred, error_details
            if evt.reason == speechsdk.CancellationReason.Error:
                error_occurred = True
                error_details = evt.error_details
            done = True

        def stop_continuous(evt):
            nonlocal done
            done = True

        speech_recognizer.recognized.connect(handle_final_result)
        speech_recognizer.session_stopped.connect(stop_continuous)
        speech_recognizer.canceled.connect(handle_canceled)

        speech_recognizer.start_continuous_recognition()

        timeout = max(duration + 10, 60)
        start_time = time.time()
        
        while not done:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                break

        speech_recognizer.stop_continuous_recognition()
        
        if error_occurred:
            speech_recognizer = None
            audio_config = None
            safe_delete_file(temp_filename, request_id)
            return jsonify({
                'success': False,
                'error': f'Recognition error: {error_details}',
                'request_id': request_id
            }), 500

        speech_recognizer = None
        audio_config = None

        safe_delete_file(temp_filename, request_id)

        if all_results:
            full_text = " ".join(all_results)
            return jsonify({
                'success': True,
                'text': full_text,
                'language': language,
                'duration': duration,
                'segments': all_segments,
                'provider': 'Azure Speech Services',
                'method': 'continuous_recognition',
                'request_id': request_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Recognition failed',
                'request_id': request_id
            }), 400

    except Exception as e:
        safe_delete_file(temp_filename, request_id)
            
        print(f"[{request_id}] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'request_id': request_id
        }), 500


@app.route('/')
def home():
    return jsonify({
        'message': 'Azure Speech-to-Text API',
        'status': 'running',
        'provider': 'Azure Cognitive Services',
        'default_language': 'en-IN (Indian English)',
        'supported_formats': ['WAV', 'MP3', 'OGG', 'M4A', 'FLAC', 'AAC', 'WebM'],
        'features': [
            'Supports audio files of any duration',
            'Automatic format conversion',
            'Continuous recognition for complete transcription',
            'Word-level timestamps available',
            'Concurrent request support (multiple users)'
        ],
        'endpoints': {
            'health': '/health',
            'transcribe': '/transcribe',
            'transcribe_with_timestamps': '/transcribe-with-timestamps'
        }
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("\n" + "="*60)
    print("🎤 AZURE SPEECH-TO-TEXT API SERVER")
    print("="*60)
    if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "your_azure_key_here":
        print("⚠️ WARNING: Please add your Azure Speech API key!")
    else:
        print("✅ Azure Speech Key: Configured")
        print(f"✅ Azure Region: {AZURE_REGION}")
    print(f"✅ Default Language: en-IN (Indian English)")
    print(f"✅ Concurrent Requests: Supported (up to 10 simultaneous)")
    print(f"\n🌐 Server: http://0.0.0.0:{port}")
    print("🎵 Supported: WAV, MP3, OGG, M4A, FLAC, AAC, WebM")
    print("⏱️  Recognition: Continuous (captures complete audio)")
    print("🔄 Multi-user: Thread-safe with unique request IDs")
    print("="*60 + "\n")
    
    # Run with threading enabled for concurrent requests
    app.run(debug=True, port=port, host='0.0.0.0', threaded=True)