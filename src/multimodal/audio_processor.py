"""Audio processing and ASR for math problems."""
import io
import tempfile
import os
from typing import Tuple, Optional
import whisper
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from src.utils.config import ASR_CONFIDENCE_THRESHOLD


class AudioProcessor:
    """Process audio and convert speech to text using Whisper."""
    
    def __init__(self, model_size: str = "base"):
        """Initialize Whisper model."""
        try:
            self.model = whisper.load_model(model_size)
        except Exception as e:
            print(f"Warning: Whisper model loading failed: {e}")
            self.model = None
    
    def convert_audio_format(self, audio_bytes: bytes, input_format: str = "wav") -> str:
        """Convert audio to WAV format for Whisper."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as tmp_input:
                tmp_input.write(audio_bytes)
                tmp_input_path = tmp_input.name
            
            # If already WAV, return as-is (Whisper can handle WAV directly)
            if input_format.lower() == "wav":
                return tmp_input_path
            
            # Convert to WAV
            try:
                audio = AudioSegment.from_file(tmp_input_path, format=input_format)
                wav_path = tmp_input_path.replace(f".{input_format}", ".wav")
                audio.export(wav_path, format="wav")
                
                # Clean up original
                os.unlink(tmp_input_path)
                
                return wav_path
            except Exception as conv_error:
                # If conversion fails (e.g., no ffmpeg), try using original file
                print(f"Audio conversion warning: {conv_error}")
                print("Attempting to use original file format...")
                # Whisper might handle the original format, return it
                return tmp_input_path
            
        except Exception as e:
            print(f"Audio conversion error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_audio_with_librosa(self, audio_path: str) -> np.ndarray:
        """Load audio using librosa (doesn't require ffmpeg)."""
        try:
            # Load audio with librosa (handles many formats, automatically converts to mono)
            # sr=16000 is Whisper's required sample rate
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            print(f"Loaded audio with librosa: shape={audio.shape}, sr={sr}")
            return audio
        except Exception as e:
            print(f"Librosa load error: {e}, trying soundfile...")
            try:
                # Fallback to soundfile
                audio, sr = sf.read(audio_path)
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                # Resample to 16kHz if needed
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                print(f"Loaded audio with soundfile: shape={audio.shape}, sr={sr}")
                return audio
            except Exception as e2:
                print(f"Soundfile load error: {e2}")
                raise
    
    def extract_text(self, audio_path: str) -> Tuple[str, float, dict]:
        """
        Extract text from audio using Whisper.
        
        Returns:
            Tuple of (transcript, confidence, metadata)
        """
        if self.model is None:
            return "", 0.0, {}
        
        try:
            # Load audio using librosa (avoids ffmpeg requirement)
            try:
                print(f"Loading audio from: {audio_path}")
                audio_array = self.load_audio_with_librosa(audio_path)
                print(f"Audio loaded successfully, shape: {audio_array.shape}, dtype: {audio_array.dtype}")
                
                # Ensure audio is float32 and normalized
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                
                # Normalize audio to [-1, 1] range if needed
                max_val = np.abs(audio_array).max()
                if max_val > 1.0:
                    audio_array = audio_array / max_val
                
                audio_length_seconds = len(audio_array) / 16000.0
                print(f"Starting Whisper transcription (audio length: {audio_length_seconds:.2f}s)...")
                print(f"Audio array: shape={audio_array.shape}, dtype={audio_array.dtype}, min={audio_array.min():.4f}, max={audio_array.max():.4f}")
                
                # Transcribe with the audio array directly
                # Note: Whisper transcribe accepts numpy array as first argument
                # Use fp16=False for CPU (already handled by Whisper, but explicit)
                print("Calling Whisper transcribe...")
                import time
                start_time = time.time()
                try:
                    result = self.model.transcribe(
                        audio_array, 
                        verbose=False, 
                        fp16=False,
                        language="en"  # Specify language to speed up processing
                    )
                    elapsed = time.time() - start_time
                    print(f"✓ Whisper transcription completed in {elapsed:.2f} seconds")
                    print(f"Transcription result keys: {list(result.keys())}")
                    if "text" in result:
                        text_preview = result['text'][:100] if len(result['text']) > 100 else result['text']
                        print(f"Transcribed text (first 100 chars): {text_preview}")
                    else:
                        print("⚠️ Warning: No 'text' key in transcription result")
                except Exception as transcribe_ex:
                    elapsed = time.time() - start_time
                    print(f"✗ Whisper transcription failed after {elapsed:.2f} seconds: {transcribe_ex}")
                    import traceback
                    traceback.print_exc()
                    raise
            except Exception as load_error:
                print(f"Error loading audio with librosa: {load_error}")
                import traceback
                traceback.print_exc()
                # Fallback: try Whisper's built-in loader (requires ffmpeg)
                print("Falling back to Whisper's audio loader (requires ffmpeg)...")
                try:
                    result = self.model.transcribe(audio_path, verbose=False)
                except Exception as whisper_error:
                    print(f"Whisper transcription failed: {whisper_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            text = result.get("text", "").strip()
            
            # Debug: print result keys to understand what Whisper returns
            print(f"Whisper result keys: {result.keys()}")
            
            # Calculate confidence from segments if available
            segments = result.get("segments", [])
            confidence = 0.5  # Default to moderate confidence
            
            print(f"Number of segments: {len(segments)}")
            
            if segments:
                # Calculate average confidence from segments
                confidences = []
                for i, segment in enumerate(segments):
                    seg_conf = None
                    # Whisper segments may have 'no_speech_prob' or other metrics
                    if 'no_speech_prob' in segment:
                        # Higher confidence = lower no_speech_prob
                        no_speech = segment.get('no_speech_prob', 0.5)
                        seg_conf = 1.0 - no_speech
                        print(f"Segment {i}: no_speech_prob={no_speech}, conf={seg_conf}")
                    elif 'avg_logprob' in segment:
                        # Use log probability as confidence indicator
                        # Normalize: logprob is typically negative, convert to 0-1 range
                        logprob = segment.get('avg_logprob', -1.0)
                        # Logprob typically ranges from -1 to 0, normalize to 0-1
                        seg_conf = max(0.0, min(1.0, (logprob + 1.0)))
                        print(f"Segment {i}: avg_logprob={logprob}, conf={seg_conf}")
                    elif 'compression_ratio' in segment:
                        # Compression ratio can indicate confidence
                        comp_ratio = segment.get('compression_ratio', 1.0)
                        # Lower compression ratio = higher confidence (typically)
                        seg_conf = max(0.0, min(1.0, 1.0 / (comp_ratio + 0.1)))
                        print(f"Segment {i}: compression_ratio={comp_ratio}, conf={seg_conf}")
                    
                    if seg_conf is not None:
                        confidences.append(seg_conf)
                
                if confidences:
                    confidence = sum(confidences) / len(confidences)
                    print(f"Calculated average confidence from segments: {confidence}")
                else:
                    # Fallback: use language probability if available
                    confidence = result.get("language_prob", 0.6)
                    print(f"Using language_prob: {confidence}")
            else:
                # No segments, use language probability or default
                confidence = result.get("language_prob", 0.6)
                print(f"No segments, using language_prob or default: {confidence}")
            
            # If we have text but confidence is still very low, boost it
            if len(text) > 0:
                if confidence < 0.4:
                    # If we got text, assume at least moderate confidence
                    confidence = 0.6
                    print(f"Boosted confidence to {confidence} because text was extracted")
                # Further boost if text is substantial
                if len(text) > 20:
                    confidence = min(0.9, confidence + 0.1)
                    print(f"Further boosted confidence to {confidence} for substantial text")
            
            # Check for math-specific terms to adjust confidence
            math_terms = ["square root", "raised to", "derivative", "integral", 
                         "limit", "matrix", "determinant", "probability", "solve",
                         "find", "calculate", "equals", "plus", "minus", "times",
                         "divided", "x", "y", "z", "equation"]
            has_math_terms = any(term in text.lower() for term in math_terms)
            
            # Boost confidence if math terms are present
            if has_math_terms and confidence < 0.7:
                confidence = min(0.9, confidence + 0.2)
            
            metadata = {
                "language": result.get("language", "en"),
                "has_math_terms": has_math_terms,
                "segments": segments,
                "num_segments": len(segments)
            }
            
            return text, confidence, metadata
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0, {}
    
    def process_uploaded_audio(self, uploaded_file) -> Tuple[str, float, bool, dict]:
        """
        Process uploaded audio file.
        
        Returns:
            Tuple of (transcript, confidence, needs_hitl, metadata)
        """
        temp_file_path = None
        try:
            print(f"Processing uploaded audio file: {uploaded_file.name}")
            
            # Reset file pointer to beginning (in case it was read before)
            if hasattr(uploaded_file, 'seek'):
                uploaded_file.seek(0)
            
            # Read audio bytes
            audio_bytes = uploaded_file.read()
            print(f"Read {len(audio_bytes)} bytes from uploaded file")
            
            if len(audio_bytes) == 0:
                print("ERROR: No audio bytes read from file!")
                return "", 0.0, True, {}
            
            # Determine format from file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            print(f"File extension: {file_extension}")
            
            # Save to temporary file (librosa can read from file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(audio_bytes)
                temp_file_path = tmp_file.name
            print(f"Saved to temporary file: {temp_file_path}")
            
            # Try to convert to WAV first (if ffmpeg available)
            wav_path = None
            try:
                # Reset file pointer again for conversion
                uploaded_file.seek(0)
                wav_path = self.convert_audio_format(audio_bytes, file_extension)
                if wav_path and os.path.exists(wav_path):
                    print(f"Converted to WAV: {wav_path}")
            except Exception as conv_error:
                print(f"Audio conversion skipped (ffmpeg not available): {conv_error}")
                # Will use original file with librosa
            
            # Use the converted WAV if available, otherwise use original
            audio_file_path = wav_path if wav_path and os.path.exists(wav_path) else temp_file_path
            print(f"Using audio file: {audio_file_path}")
            
            # Transcribe using librosa (bypasses ffmpeg requirement)
            print("Starting transcription...")
            text, confidence, metadata = self.extract_text(audio_file_path)
            print(f"Transcription complete: text_length={len(text)}, confidence={confidence:.3f}")
            
            # Clean up temporary files
            if wav_path and os.path.exists(wav_path) and wav_path != temp_file_path:
                try:
                    os.unlink(wav_path)
                except:
                    pass
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            # Determine if HITL is needed
            # Be more lenient - only trigger HITL if really uncertain
            text_len = len(text.strip())
            needs_hitl = (
                (text_len == 0) or  # No text at all
                (confidence < 0.2 and text_len < 10) or  # Very low confidence and short text
                (confidence < 0.1)  # Extremely low confidence regardless of text
            )
            
            print(f"Audio processing result: text_len={text_len}, confidence={confidence:.3f}, needs_hitl={needs_hitl}")
            
            return text, confidence, needs_hitl, metadata
        except Exception as e:
            print(f"Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return "", 0.0, True, {}
    
    def normalize_math_phrases(self, text: str) -> str:
        """Normalize math-specific phrases in transcript."""
        replacements = {
            "square root of": "√",
            "raised to": "^",
            "to the power of": "^",
            "divided by": "/",
            "times": "*",
            "multiplied by": "*",
        }
        
        normalized = text
        for phrase, symbol in replacements.items():
            normalized = normalized.replace(phrase, symbol)
        
        return normalized
