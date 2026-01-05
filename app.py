from flask import Flask, request, jsonify
import azure.cognitiveservices.speech as speechsdk
import os
import tempfile
import time
import uuid
import subprocess
import shutil

app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

# Azure Speech Service credentials
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY', 'e99111621cee4a7ea292xxxxxxxxxxxx')
AZURE_REGION = os.getenv('AZURE_REGION', 'centralindia')

# Check if ffmpeg is available
FFMPEG_PATH = shutil.which('ffmpeg')


def convert_to_wav_ffmpeg(input_path, request_id):
    """Convert audio to WAV using ffmpeg"""
    try:
        print(f"[{request_id}] 🔄 Converting audio with ffmpeg")
        
        # Create output temp file
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', prefix=f'converted_{request_id}_')
        output_path = output_file.name
        output_file.close()
        
        # FFmpeg command to convert to 16kHz, 16-bit, mono WAV
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',          # Sample rate 16kHz
            '-ac', '1',              # Mono
            '-sample_fmt', 's16',    # 16-bit
            '-y',                    # Overwrite output
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr.decode()}")
        
        print(f"[{request_id}] ✅ Converted to WAV: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception("Audio conversion timeout (file too large or corrupted)")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Please install ffmpeg.")
    except Exception as e:
        raise Exception(f"Conversion failed: {str(e)}")


def save_uploaded_file(audio_file, filename, request_id):
    """Save uploaded audio file temporarily"""
    try:
        print(f"[{request_id}] 💾 Saving audio: {filename}")
        
        # Create unique temp file for this request
        file_ext = os.path.splitext(filename)[1] or '.audio'
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_ext, 
            prefix=f'upload_{request_id}_'
        )
        audio_file.save(temp_file.name)
        temp_file.close()
        
        print(f"[{request_id}] ✅ Audio saved: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"[{request_id}] ❌ Save error: {str(e)}")
        raise


def safe_delete_file(filename, request_id):
    """Safely delete a file with retry logic"""
    if filename and os.path.exists(filename):
        for attempt in range(3):
            try:
                os.unlink(filename)
                print(f"[{request_id}] 🗑️ Deleted: {os.path.basename(filename)}")
                return True
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.2)
                else:
                    print(f"[{request_id}] ⚠️ Could not delete: {filename}")
                    return False
    return True


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    ffmpeg_status = "available" if FFMPEG_PATH else "not found"
    return jsonify({
        'status': 'healthy',
        'service': 'Azure Speech-to-Text API',
        'version': '1.0',
        'provider': 'Azure Cognitive Services',
        'default_language': 'en-IN',
        'ffmpeg': ffmpeg_status,
        'concurrent_support': True
    }), 200


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Speech-to-Text endpoint - handles all audio durations"""
    request_id = str(uuid.uuid4())[:8]
    temp_input = None
    temp_wav = None
    
    try:
        if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "your_azure_key_here":
            return jsonify({'error': 'Azure Speech API key not configured'}), 500

        if not FFMPEG_PATH:
            return jsonify({'error': 'FFmpeg not installed on server'}), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        language = request.form.get('language', 'en-IN')

        print(f"\n[{request_id}] {'='*50}")
        print(f"[{request_id}] 📝 Transcription Request:")
        print(f"[{request_id}]    File: {audio_file.filename}")
        print(f"[{request_id}]    Language: {language}")
        print(f"[{request_id}] {'='*50}\n")

        # Step 1: Save uploaded file
        try:
            temp_input = save_uploaded_file(audio_file, audio_file.filename, request_id)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to save file: {str(e)}',
                'request_id': request_id
            }), 400

        # Step 2: Convert to Azure-compatible WAV
        try:
            temp_wav = convert_to_wav_ffmpeg(temp_input, request_id)
            # Clean up input file after conversion
            safe_delete_file(temp_input, request_id)
            temp_input = None
        except Exception as e:
            safe_delete_file(temp_input, request_id)
            return jsonify({
                'success': False,
                'error': f'Audio conversion failed: {str(e)}',
                'message': 'Please upload a valid audio file',
                'request_id': request_id
            }), 400

        # Step 3: Configure Azure Speech Service
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        speech_config.speech_recognition_language = language

        # Create audio configuration
        audio_config = speechsdk.AudioConfig(filename=temp_wav)

        print(f"[{request_id}] 🔄 Starting continuous recognition")
        
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
            print(f"[{request_id}] 🔄 Recognizing: {evt.result.text}")

        def handle_final_result(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                all_results.append(evt.result.text)
                print(f"[{request_id}] 📝 Recognized: {evt.result.text}")
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                print(f"[{request_id}] ⚠️ No match")

        def handle_canceled(evt):
            nonlocal done, error_occurred, error_details
            print(f"[{request_id}] ⚠️ Canceled: {evt.reason}")
            if evt.reason == speechsdk.CancellationReason.Error:
                error_occurred = True
                error_details = evt.error_details
                print(f"[{request_id}] ❌ Error: {evt.error_details}")
            done = True

        def stop_continuous(evt):
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

        # Wait with timeout
        timeout = 180  # 3 minutes
        start_time = time.time()
        
        while not done:
            time.sleep(0.1)
            if time.time() - start_time > timeout:
                print(f"[{request_id}] ⚠️ Timeout after {timeout}s")
                break

        # Stop recognition
        speech_recognizer.stop_continuous_recognition()
        
        # Clean up
        speech_recognizer = None
        audio_config = None
        time.sleep(0.3)  # Give time for file handles to release
        safe_delete_file(temp_wav, request_id)
        
        # Check for errors
        if error_occurred:
            return jsonify({
                'success': False,
                'error': f'Recognition error: {error_details}',
                'request_id': request_id
            }), 500

        if all_results:
            full_text = " ".join(all_results)
            print(f"[{request_id}] ✅ Transcription: {full_text}\n")
            
            return jsonify({
                'success': True,
                'text': full_text,
                'language': language,
                'provider': 'Azure Speech Services',
                'request_id': request_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'No speech recognized',
                'message': 'Audio may be silent or unclear',
                'request_id': request_id
            }), 400

    except Exception as e:
        # Clean up any remaining temp files
        safe_delete_file(temp_input, request_id)
        safe_delete_file(temp_wav, request_id)
        
        print(f"[{request_id}] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Transcription failed',
            'request_id': request_id
        }), 500


@app.route('/transcribe-with-timestamps', methods=['POST'])
def transcribe_with_timestamps():
    """Speech-to-Text with timestamps"""
    request_id = str(uuid.uuid4())[:8]
    temp_input = None
    temp_wav = None
    
    try:
        if not AZURE_SPEECH_KEY or AZURE_SPEECH_KEY == "your_azure_key_here":
            return jsonify({'error': 'Azure API key not configured'}), 500

        if not FFMPEG_PATH:
            return jsonify({'error': 'FFmpeg not installed'}), 500

        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        language = request.form.get('language', 'en-IN')

        # Save and convert
        temp_input = save_uploaded_file(audio_file, audio_file.filename, request_id)
        temp_wav = convert_to_wav_ffmpeg(temp_input, request_id)
        safe_delete_file(temp_input, request_id)

        # Configure Azure
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_REGION
        )
        speech_config.speech_recognition_language = language
        speech_config.request_word_level_timestamps()
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        audio_config = speechsdk.AudioConfig(filename=temp_wav)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        all_results = []
        all_segments = []
        done = False

        def handle_result(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                import json
                detailed = json.loads(evt.result.json)
                all_results.append(evt.result.text)
                if 'NBest' in detailed and detailed['NBest']:
                    all_segments.append(detailed['NBest'][0])

        def stop_cb(evt):
            nonlocal done
            done = True

        speech_recognizer.recognized.connect(handle_result)
        speech_recognizer.session_stopped.connect(stop_cb)
        speech_recognizer.canceled.connect(stop_cb)

        speech_recognizer.start_continuous_recognition()

        timeout = 180
        start = time.time()
        while not done and (time.time() - start < timeout):
            time.sleep(0.1)

        speech_recognizer.stop_continuous_recognition()
        speech_recognizer = None
        audio_config = None
        time.sleep(0.3)
        safe_delete_file(temp_wav, request_id)

        if all_results:
            return jsonify({
                'success': True,
                'text': " ".join(all_results),
                'language': language,
                'segments': all_segments,
                'request_id': request_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'No speech recognized',
                'request_id': request_id
            }), 400

    except Exception as e:
        safe_delete_file(temp_input, request_id)
        safe_delete_file(temp_wav, request_id)
        print(f"[{request_id}] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e), 'request_id': request_id}), 500


@app.route('/')
def home():
    ffmpeg_status = "✅ Installed" if FFMPEG_PATH else "❌ Not Found"
    return jsonify({
        'message': 'Azure Speech-to-Text API',
        'status': 'running',
        'provider': 'Azure Cognitive Services',
        'default_language': 'en-IN (Indian English)',
        'ffmpeg': ffmpeg_status,
        'supported_formats': ['WAV', 'MP3', 'OGG', 'M4A', 'FLAC', 'AAC', 'OPUS', 'WebM'],
        'features': [
            'Handles audio of ANY duration',
            'FFmpeg audio conversion',
            'Continuous recognition',
            'Concurrent requests supported'
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
        print("⚠️ WARNING: Azure Speech API key not configured")
    else:
        print("✅ Azure Speech Key: Configured")
        print(f"✅ Azure Region: {AZURE_REGION}")
    
    if FFMPEG_PATH:
        print(f"✅ FFmpeg: {FFMPEG_PATH}")
    else:
        print("❌ FFmpeg: NOT FOUND - Please install ffmpeg!")
    
    print(f"✅ Default Language: en-IN (Indian English)")
    print(f"\n🌐 Server: http://0.0.0.0:{port}")
    print("🎵 All audio formats supported")
    print("⏱️  Handles ANY duration audio")
    print("="*60 + "\n")
    
    app.run(debug=True, port=port, host='0.0.0.0', threaded=True)
