# import os
# import tempfile
# from flask import Flask, request, jsonify
# import numpy as np  # Make sure numpy is installed
# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration

# app = Flask(__name__)

# # Define model path
# MODEL_PATH = "/root/Baibhav/hackathon/Medical-RAG/models/whisper-small"

# # Check if model exists, if not download it
# if not os.path.exists(MODEL_PATH):
#     os.makedirs(MODEL_PATH, exist_ok=True)
#     print(f"Model directory created at {MODEL_PATH}")
#     # We'll load from HF and save to this directory
#     processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#     model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
#     processor.save_pretrained(MODEL_PATH)
#     model.save_pretrained(MODEL_PATH)
#     print(f"Model downloaded and saved to {MODEL_PATH}")
# else:
#     print(f"Using existing model at {MODEL_PATH}")

# # Verify numpy is available
# try:
#     import numpy as np
#     print(f"NumPy version: {np.__version__}")
# except ImportError:
#     print("NumPy is not available. Installing NumPy...")
#     import subprocess
#     subprocess.check_call(["pip", "install", "numpy"])
#     import numpy as np
#     print(f"NumPy installed successfully. Version: {np.__version__}")

# # Load model and processor
# print("Loading Whisper model...")
# processor = WhisperProcessor.from_pretrained(MODEL_PATH)
# model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# # Use GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)
# print(f"Model loaded successfully and running on {device}")

# @app.route('/health', methods=['GET'])
# def health_check():
#     return jsonify({
#         "status": "ok",
#         "model": "whisper-small",
#         "device": device
#     })

# @app.route('/transcribe', methods=['POST'])
# def transcribe_audio():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400
    
#     audio_file = request.files['file']
    
#     # Get parameters from the request
#     language = request.form.get('language', 'english')
#     task = request.form.get('task', 'transcribe')  # 'transcribe' or 'translate'
#     return_timestamps = request.form.get('return_timestamps', 'false').lower() == 'true'
    
#     # Save uploaded file to a temporary location
#     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
#         audio_file.save(temp_audio.name)
#         temp_filename = temp_audio.name
    
#     try:
#         # Process the audio file
#         try:
#             # Verify numpy is available in this context
#             import numpy
            
#             import librosa
#             audio_array, sampling_rate = librosa.load(temp_filename, sr=16000)
#         except ImportError as e:
#             print(f"Import error: {e}")
#             # Fallback to use soundfile if librosa is not available
#             try:
#                 import soundfile as sf
#                 audio_array, sampling_rate = sf.read(temp_filename)
#                 if sampling_rate != 16000:
#                     # Need to resample
#                     import resampy
#                     audio_array = resampy.resample(audio_array, sampling_rate, 16000)
#                     sampling_rate = 16000
#             except ImportError as sf_error:
#                 print(f"Both librosa and soundfile failed: {sf_error}")
#                 return jsonify({"error": f"Audio processing libraries not available: {e}, {sf_error}"}), 500
        
#         # Prepare the model inputs
#         input_features = processor(
#             audio_array, 
#             sampling_rate=sampling_rate, 
#             return_tensors="pt"
#         ).input_features.to(device)
        
#         # Set decoder ids based on language and task
#         forced_decoder_ids = processor.get_decoder_prompt_ids(
#             language=language, 
#             task=task
#         )
        
#         # Generate the transcription
#         with torch.no_grad():
#             if return_timestamps:
#                 # For timestamp generation
#                 predicted_ids = model.generate(
#                     input_features, 
#                     forced_decoder_ids=forced_decoder_ids,
#                     return_timestamps=True
#                 )
#             else:
#                 predicted_ids = model.generate(
#                     input_features, 
#                     forced_decoder_ids=forced_decoder_ids
#                 )
        
#         # Decode the prediction
#         transcription = processor.batch_decode(
#             predicted_ids, 
#             skip_special_tokens=True
#         )[0]
        
#         return jsonify({
#             "transcription": transcription,
#             "language": language,
#             "task": task
#         })
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"Error in transcription: {error_details}")
#         return jsonify({"error": str(e), "details": error_details}), 500
    
#     finally:
#         # Clean up the temporary file
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)

# @app.route('/transcribe/long', methods=['POST'])
# def transcribe_long_audio():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file provided"}), 400
    
#     audio_file = request.files['file']
    
#     # Get parameters from the request
#     language = request.form.get('language', 'english')
#     task = request.form.get('task', 'transcribe')  # 'transcribe' or 'translate'
#     return_timestamps = request.form.get('return_timestamps', 'false').lower() == 'true'
#     chunk_length_s = int(request.form.get('chunk_length_s', '30'))
#     batch_size = int(request.form.get('batch_size', '8'))
    
#     # Save uploaded file to a temporary location
#     with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
#         audio_file.save(temp_audio.name)
#         temp_filename = temp_audio.name
    
#     try:
#         # Verify numpy is available in this context
#         import numpy
        
#         # Process the long audio file with chunking
#         from transformers import pipeline
        
#         pipe = pipeline(
#             "automatic-speech-recognition",
#             model=MODEL_PATH,
#             chunk_length_s=chunk_length_s,
#             device=device,
#         )
        
#         # Configure pipeline
#         if language != "english" or task != "transcribe":
#             pipe.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
#                 language=language, 
#                 task=task
#             )
        
#         # Process the audio
#         if return_timestamps:
#             result = pipe(temp_filename, batch_size=batch_size, return_timestamps=True)
#             return jsonify({
#                 "chunks": result["chunks"],
#                 "language": language,
#                 "task": task
#             })
#         else:
#             result = pipe(temp_filename, batch_size=batch_size)
#             return jsonify({
#                 "transcription": result["text"],
#                 "language": language,
#                 "task": task
#             })
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"Error in long transcription: {error_details}")
#         return jsonify({"error": str(e), "details": error_details}), 500
    
#     finally:
#         # Clean up the temporary file
#         if os.path.exists(temp_filename):
#             os.remove(temp_filename)

# @app.route('/languages', methods=['GET'])
# def get_languages():
#     # Return list of supported languages
#     languages = [
#         "english", "chinese", "german", "spanish", "russian", "korean", 
#         "french", "japanese", "portuguese", "turkish", "polish", "catalan", 
#         "dutch", "arabic", "swedish", "italian", "indonesian", "hindi", 
#         "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay", 
#         "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", 
#         "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin", 
#         "maori", "malayalam", "welsh", "slovak", "telugu", "persian", 
#         "latvian", "bengali", "serbian", "azerbaijani", "slovenian", 
#         "kannada", "estonian", "macedonian", "breton", "basque", "icelandic", 
#         "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian", 
#         "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer", 
#         "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", 
#         "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish", 
#         "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen", 
#         "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan", 
#         "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", 
#         "hausa", "bashkir", "javanese", "sundanese"
#     ]
#     return jsonify({"languages": languages})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)


import os
import tempfile
from flask import Flask, request, jsonify
import numpy as np  # Make sure numpy is installed
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

app = Flask(__name__)

# Define model path
MODEL_PATH = "/root/Baibhav/hackathon/Medical-RAG/models/whisper-small"

# Check if model exists, if not download it
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)
    print(f"Model directory created at {MODEL_PATH}")
    # We'll load from HF and save to this directory
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)
    print(f"Model downloaded and saved to {MODEL_PATH}")
else:
    print(f"Using existing model at {MODEL_PATH}")

# Verify numpy is available
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy is not available. Installing NumPy...")
    import subprocess
    subprocess.check_call(["pip", "install", "numpy"])
    import numpy as np
    print(f"NumPy installed successfully. Version: {np.__version__}")

# Load model and processor
print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

# Use CUDA:1 specifically
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    device = torch.device("cuda:1")
    print(f"Using CUDA device 1: {torch.cuda.get_device_name(1)}")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Requested CUDA:1 but using {device} instead")

model = model.to(device)
print(f"Model loaded successfully and running on {device}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model": "whisper-small",
        "device": str(device)
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    audio_file = request.files['file']
    
    # Get parameters from the request
    language = request.form.get('language', 'english')
    task = request.form.get('task', 'transcribe')  # 'transcribe' or 'translate'
    return_timestamps = request.form.get('return_timestamps', 'false').lower() == 'true'
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_file.save(temp_audio.name)
        temp_filename = temp_audio.name
    
    try:
        # Process the audio file
        try:
            # Verify numpy is available in this context
            import numpy
            
            import librosa
            audio_array, sampling_rate = librosa.load(temp_filename, sr=16000)
        except ImportError as e:
            print(f"Import error: {e}")
            # Fallback to use soundfile if librosa is not available
            try:
                import soundfile as sf
                audio_array, sampling_rate = sf.read(temp_filename)
                if sampling_rate != 16000:
                    # Need to resample
                    import resampy
                    audio_array = resampy.resample(audio_array, sampling_rate, 16000)
                    sampling_rate = 16000
            except ImportError as sf_error:
                print(f"Both librosa and soundfile failed: {sf_error}")
                return jsonify({"error": f"Audio processing libraries not available: {e}, {sf_error}"}), 500
        
        # Prepare the model inputs
        input_features = processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # Set decoder ids based on language and task
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, 
            task=task
        )
        
        # Generate the transcription
        with torch.no_grad():
            if return_timestamps:
                # For timestamp generation
                predicted_ids = model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids,
                    return_timestamps=True
                )
            else:
                predicted_ids = model.generate(
                    input_features, 
                    forced_decoder_ids=forced_decoder_ids
                )
        
        # Decode the prediction
        transcription = processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return jsonify({
            "transcription": transcription,
            "language": language,
            "task": task
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in transcription: {error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.route('/transcribe/long', methods=['POST'])
def transcribe_long_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    audio_file = request.files['file']
    
    # Get parameters from the request
    language = request.form.get('language', 'english')
    task = request.form.get('task', 'transcribe')  # 'transcribe' or 'translate'
    return_timestamps = request.form.get('return_timestamps', 'false').lower() == 'true'
    chunk_length_s = int(request.form.get('chunk_length_s', '30'))
    batch_size = int(request.form.get('batch_size', '8'))
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_file.save(temp_audio.name)
        temp_filename = temp_audio.name
    
    try:
        # Verify numpy is available in this context
        import numpy
        
        # Process the long audio file with chunking
        from transformers import pipeline
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_PATH,
            chunk_length_s=chunk_length_s,
            device=device,
        )
        
        # Configure pipeline
        if language != "english" or task != "transcribe":
            pipe.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language, 
                task=task
            )
        
        # Process the audio
        if return_timestamps:
            result = pipe(temp_filename, batch_size=batch_size, return_timestamps=True)
            return jsonify({
                "chunks": result["chunks"],
                "language": language,
                "task": task
            })
        else:
            result = pipe(temp_filename, batch_size=batch_size)
            return jsonify({
                "transcription": result["text"],
                "language": language,
                "task": task
            })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in long transcription: {error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.route('/languages', methods=['GET'])
def get_languages():
    # Return list of supported languages
    languages = [
        "english", "chinese", "german", "spanish", "russian", "korean", 
        "french", "japanese", "portuguese", "turkish", "polish", "catalan", 
        "dutch", "arabic", "swedish", "italian", "indonesian", "hindi", 
        "finnish", "vietnamese", "hebrew", "ukrainian", "greek", "malay", 
        "czech", "romanian", "danish", "hungarian", "tamil", "norwegian", 
        "thai", "urdu", "croatian", "bulgarian", "lithuanian", "latin", 
        "maori", "malayalam", "welsh", "slovak", "telugu", "persian", 
        "latvian", "bengali", "serbian", "azerbaijani", "slovenian", 
        "kannada", "estonian", "macedonian", "breton", "basque", "icelandic", 
        "armenian", "nepali", "mongolian", "bosnian", "kazakh", "albanian", 
        "swahili", "galician", "marathi", "punjabi", "sinhala", "khmer", 
        "shona", "yoruba", "somali", "afrikaans", "occitan", "georgian", 
        "belarusian", "tajik", "sindhi", "gujarati", "amharic", "yiddish", 
        "lao", "uzbek", "faroese", "haitian creole", "pashto", "turkmen", 
        "nynorsk", "maltese", "sanskrit", "luxembourgish", "myanmar", "tibetan", 
        "tagalog", "malagasy", "assamese", "tatar", "hawaiian", "lingala", 
        "hausa", "bashkir", "javanese", "sundanese"
    ]
    return jsonify({"languages": languages})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=False)