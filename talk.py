import os
import torch
import argparse
import pyaudio
import wave
from zipfile import ZipFile
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import openai
from openai import OpenAI
import os
import time
import speech_recognition as sr
import whisper
import threading
import queue
import numpy as np
from collections import deque
import subprocess
import warnings

# Suppress ALSA warnings
os.environ['ALSA_PCM_CARD'] = '0'
os.environ['ALSA_PCM_DEVICE'] = '0'

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Global variables for thread control
stop_listening = threading.Event()
stop_speaking = threading.Event()
audio_queue = queue.Queue()
is_speaking = threading.Event()
user_speaking = threading.Event()
audio_playback_active = threading.Event()

# Voice Activity Detection parameters
BASE_SILENCE_THRESHOLD = 300   # Base threshold - much lower
SILENCE_DURATION = 2.0         # Silence duration
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Echo cancellation parameters
ECHO_SUPPRESSION_FACTOR = 3.0  # High suppression when AI is speaking
MIN_SPEECH_DURATION = 0.2      # Faster response for interruptions
BACKGROUND_NOISE_SAMPLES = 30   # Samples to calculate background noise
MAX_NOISE_MULTIPLIER = 3.0     # Maximum multiplier for background noise
INTERRUPTION_THRESHOLD_MULTIPLIER = 1.8  # Balanced threshold for interruptions
INTERRUPTION_MIN_DURATION = 0.6  # Require 0.6 seconds of sustained speech
INTERRUPTION_MIN_LEVEL = 1500   # Minimum absolute level for interruption (reduced from 2000)

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

def find_working_audio_device():
    """Find a working audio input device"""
    p = pyaudio.PyAudio()
    
    print("🔍 Scanning for audio devices...")
    
    # Get default input device
    try:
        default_device = p.get_default_input_device_info()
        device_index = default_device['index']
        device_name = default_device['name']
        print(f"✅ Found default input device: {device_name} (Index: {device_index})")
        
        # Test if the device works
        try:
            test_stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE
            )
            test_stream.close()
            p.terminate()
            return device_index
        except Exception as e:
            print(f"❌ Default device test failed: {e}")
    except Exception as e:
        print(f"❌ No default input device found: {e}")
    
    # If default doesn't work, try all devices
    print("🔍 Testing all available input devices...")
    device_count = p.get_device_count()
    
    for i in range(device_count):
        try:
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Input device
                print(f"Testing device {i}: {device_info['name']}")
                
                test_stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=CHUNK_SIZE
                )
                test_stream.close()
                print(f"✅ Device {i} works: {device_info['name']}")
                p.terminate()
                return i
        except Exception as e:
            print(f"❌ Device {i} failed: {e}")
            continue
    
    p.terminate()
    print("❌ No working audio input device found!")
    return None

# Function to play audio using PyAudio
def play_audio(file_path):
    """Use system audio player - much more reliable than PyAudio"""
    if not os.path.exists(file_path):
        print(f"ERROR: Audio file not found: {file_path}")
        return
    
    # Try different system audio players
    audio_players = [
        ['aplay', '-q', file_path],
        ['paplay', file_path],
        ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', file_path],
        ['mpg123', '-q', file_path],
        ['cvlc', '--intf', 'dummy', '--play-and-exit', '--quiet', file_path]
    ]
    
    for player_cmd in audio_players:
        try:
            result = subprocess.run(
                player_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30
            )
            if result.returncode == 0:
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            continue
    
    print("ERROR: No working audio player found.")

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
print("🤖 Loading TTS models...")
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth', weights_only=True).to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth', weights_only=True).to(device)

class AudioLevelMonitor:
    """Monitor audio levels for background noise estimation and echo detection"""
    def __init__(self, window_size=BACKGROUND_NOISE_SAMPLES):
        self.window_size = window_size
        self.background_levels = deque(maxlen=window_size)
        self.current_level = 0
        self.background_noise_level = 0
        self.is_calibrated = False
        self.calibration_samples = 0
        self.max_seen_level = 0
        
    def update(self, audio_data):
        """Update with new audio data"""
        rms = calculate_rms(audio_data)
        self.current_level = rms
        self.max_seen_level = max(self.max_seen_level, rms)
        
        # Only update background noise when not speaking
        if not is_speaking.is_set() and not user_speaking.is_set():
            self.background_levels.append(rms)
            self.calibration_samples += 1
            
            if len(self.background_levels) >= self.window_size:
                # Use a more conservative approach for background noise
                background_mean = np.mean(self.background_levels)
                background_std = np.std(self.background_levels)
                
                # Set threshold as mean + 1*std for better sensitivity
                calculated_threshold = background_mean + background_std
                
                # Ensure threshold is reasonable - between base threshold and max multiplier
                min_threshold = BASE_SILENCE_THRESHOLD
                max_threshold = min(background_mean * 2.5, self.max_seen_level * 0.2)  # More conservative
                
                self.background_noise_level = max(min_threshold, min(calculated_threshold, max_threshold))
                
                if not self.is_calibrated:
                    print(f"🎯 Calibrated - Background: {background_mean:.1f}, Threshold: {self.background_noise_level:.1f}")
                    print(f"📊 Audio levels - Current: {rms:.1f}, Max seen: {self.max_seen_level:.1f}")
                    self.is_calibrated = True
    
    def is_voice_detected(self, for_interruption=False):
        """Detect if current audio is likely human voice (not AI feedback)"""
        # Use base threshold if not calibrated
        if not self.is_calibrated:
            threshold = BASE_SILENCE_THRESHOLD
        else:
            threshold = self.background_noise_level
        
        # For interruptions, use a HIGHER threshold but not too high
        if for_interruption:
            threshold = max(
                threshold * INTERRUPTION_THRESHOLD_MULTIPLIER,
                INTERRUPTION_MIN_LEVEL,
                self.background_noise_level * 2.0  # Reduced from 2.5
            )
        # Reduce sensitivity when AI is speaking to prevent echo (but not for interruptions)
        elif is_speaking.is_set() or audio_playback_active.is_set():
            threshold *= ECHO_SUPPRESSION_FACTOR
        
        # Debug output for first few detections
        is_voice = self.current_level > threshold
        
        if self.calibration_samples < 100 and is_voice:
            print(f"🔊 Voice detected! Level: {self.current_level:.1f} > Threshold: {threshold:.1f}")
        
        # Show interruption attempts (but don't spam)
        if (is_speaking.is_set() or audio_playback_active.is_set()) and for_interruption:
            if is_voice:
                print(f"🔥 INTERRUPTION! Voice level: {self.current_level:.1f} > Threshold: {threshold:.1f}")
            elif self.current_level > self.background_noise_level * 1.2:  # Only show significant noise
                print(f"🔇 Noise detected but below interruption threshold: {self.current_level:.1f} < {threshold:.1f}")
        
        return is_voice

audio_monitor = AudioLevelMonitor()

def calculate_rms(audio_data):
    """Calculate RMS (Root Mean Square) for voice activity detection with error handling"""
    try:
        if len(audio_data) == 0:
            return 0.0
        
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio_np) == 0:
            return 0.0
        
        # Calculate RMS with safe handling of negative/zero values
        squared = audio_np.astype(np.float64) ** 2
        mean_squared = np.mean(squared)
        
        if mean_squared <= 0:
            return 0.0
        
        rms = np.sqrt(mean_squared)
        
        # Check for invalid values
        if np.isnan(rms) or np.isinf(rms):
            return 0.0
            
        return float(rms)
        
    except Exception as e:
        return 0.0

def continuous_audio_capture():
    """Continuously capture audio and detect voice activity with echo cancellation"""
    device_index = find_working_audio_device()
    if device_index is None:
        print("❌ Cannot start audio capture - no working microphone found!")
        stop_listening.set()
        return
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print("🎤 Listening... (calibrating background noise - try speaking)")
        
        # Increased buffer to 10 seconds
        audio_buffer = deque(maxlen=int(SAMPLE_RATE * 10 / CHUNK_SIZE))
        silence_chunks = 0
        speech_chunks = 0
        speech_detected = False
        debug_counter = 0  # For periodic debug output
        
        # Interruption tracking
        interruption_speech_chunks = 0
        
        while not stop_listening.is_set():
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                debug_counter += 1
                
                # Update audio level monitoring
                audio_monitor.update(data)
                
                # Periodic debug output during calibration
                if not audio_monitor.is_calibrated and debug_counter % 50 == 0:
                    print(f"📊 Current audio level: {audio_monitor.current_level:.1f} (samples: {audio_monitor.calibration_samples})")
                
                # Always check for voice activity (don't skip during playback for interruption)
                audio_buffer.append(data)
                
                # Enhanced voice activity detection
                has_voice = audio_monitor.is_voice_detected()
                
                # If AI is speaking and user speaks, interrupt immediately with higher threshold
                if (is_speaking.is_set() or audio_playback_active.is_set()):
                    interruption_detected = audio_monitor.is_voice_detected(for_interruption=True)
                    
                    # Debug: Show what thresholds are being used
                    current_level = audio_monitor.current_level
                    if audio_monitor.is_calibrated:
                        interruption_threshold = max(
                            audio_monitor.background_noise_level * INTERRUPTION_THRESHOLD_MULTIPLIER,
                            INTERRUPTION_MIN_LEVEL,
                            audio_monitor.background_noise_level * 2.5
                        )
                        
                        # Only show significant noise during AI speech
                        if current_level > audio_monitor.background_noise_level * 1.2:
                            if interruption_detected:
                                print(f"🔥 POTENTIAL INTERRUPTION: {current_level:.1f} > {interruption_threshold:.1f}")
                            else:
                                print(f"🔇 Audio below interruption threshold: {current_level:.1f} < {interruption_threshold:.1f}")
                        
                        # Show progress toward interruption
                        if interruption_detected:
                            progress = interruption_speech_chunks / min_interruption_chunks * 100
                            print(f"📈 Interruption progress: {progress:.1f}% ({interruption_speech_chunks}/{min_interruption_chunks} chunks)")
                    
                    if interruption_detected:
                        interruption_speech_chunks += 1
                        
                        # Require sustained speech for interruption
                        min_interruption_chunks = int(INTERRUPTION_MIN_DURATION * SAMPLE_RATE / CHUNK_SIZE)
                        
                        if interruption_speech_chunks >= min_interruption_chunks:
                            print("🛑 SUSTAINED USER INTERRUPTION DETECTED!")
                            stop_speaking.set()
                            user_speaking.set()
                            # Clear buffers and reset
                            audio_buffer.clear()
                            speech_detected = False
                            silence_chunks = 0
                            speech_chunks = 0
                            interruption_speech_chunks = 0
                            continue
                    else:
                        # Reset interruption counter if no voice detected
                        interruption_speech_chunks = 0
                
                # Skip normal processing if we're in the middle of playing audio
                if audio_playback_active.is_set():
                    continue
                
                if has_voice:
                    silence_chunks = 0
                    speech_chunks += 1
                    
                    # Require minimum speech duration to avoid false positives
                    min_speech_chunks = int(MIN_SPEECH_DURATION * SAMPLE_RATE / CHUNK_SIZE)
                    
                    if speech_chunks >= min_speech_chunks and not speech_detected:
                        speech_detected = True
                        user_speaking.set()
                        
                        # Stop any ongoing AI speech
                        if is_speaking.is_set():
                            stop_speaking.set()
                            print("🛑 Interrupted AI response")
                        
                        print("🗣️  User speech detected...")
                else:
                    silence_chunks += 1
                    if not speech_detected:
                        speech_chunks = 0  # Reset if we haven't detected speech yet
                
                # Check if we have enough silence to process speech
                silence_duration = silence_chunks * CHUNK_SIZE / SAMPLE_RATE
                
                if speech_detected and silence_duration >= SILENCE_DURATION:
                    print("⏸️  Processing speech...")
                    
                    # Convert audio buffer to bytes
                    audio_data = b''.join(list(audio_buffer))
                    
                    if len(audio_data) > 0:
                        # Save to temporary file for transcription
                        temp_filename = "temp_live_recording.wav"
                        try:
                            with wave.open(temp_filename, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(p.get_sample_size(FORMAT))
                                wf.setframerate(SAMPLE_RATE)
                                wf.writeframes(audio_data)
                            
                            # Add to processing queue
                            audio_queue.put(temp_filename)
                            
                        except Exception as e:
                            print(f"Error saving audio file: {e}")
                    
                    # Reset detection
                    speech_detected = False
                    silence_chunks = 0
                    speech_chunks = 0
                    user_speaking.clear()
                    audio_buffer.clear()
                    
                    if audio_monitor.is_calibrated:
                        print("🎤 Listening... (say something)")
                    else:
                        print("🎤 Listening... (calibrating - try speaking)")
                    
            except OSError as e:
                if e.errno == -9981:  # Input overflow
                    continue
                else:
                    print(f"Audio error: {e}")
                    break
                    
    except Exception as e:
        print(f"Audio capture error: {e}")
    finally:
        try:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
        except:
            pass
        p.terminate()

def process_and_play_interruptible(prompt, style, audio_file_pth):
    """Modified process_and_play function with interruption support"""
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se
    speaker_wav = audio_file_pth
    
    if not os.path.exists(audio_file_pth):
        print("ERROR - Reference audio file not found!")
        return
    
    # Check for interruption before starting
    if stop_speaking.is_set():
        return
    
    # Process text and generate audio
    try:
        print("🎙️ Generating speech...")
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
        
        if stop_speaking.is_set():
            return
            
        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')
        
        if stop_speaking.is_set():
            return
            
        save_path = f'{output_dir}/output.wav'
        
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message=encode_message
        )
        
        if stop_speaking.is_set():
            return
            
        # Set speaking flag and play audio
        is_speaking.set()
        audio_playback_active.set()
        
        # Play audio in a way that can be interrupted
        play_audio_interruptible(save_path)
        
        is_speaking.clear()
        
        # Add delay after speaking to prevent echo pickup
        time.sleep(1.0)  # Increased delay
        audio_playback_active.clear()
        
        # Reset any lingering speech detection flags
        user_speaking.clear()
        stop_speaking.clear()
        
        print("🎤 Ready for next input...")
        
    except Exception as e:
        print(f"Error during audio generation: {e}")
        is_speaking.clear()
        audio_playback_active.clear()

def play_audio_interruptible(file_path):
    """Play audio but stop if interrupted"""
    if not os.path.exists(file_path):
        print(f"ERROR: Audio file not found: {file_path}")
        return
    
    print(f"🔊 AI speaking...")
    
    # Try different system audio players with interruption capability
    audio_players = [
        ['aplay', '-q', file_path],
        ['paplay', file_path],
        ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', file_path],
        ['mpg123', '-q', file_path]
    ]
    
    for player_cmd in audio_players:
        try:
            process = subprocess.Popen(
                player_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Poll the process and check for interruption
            while process.poll() is None:
                if stop_speaking.is_set():
                    print("🛑 Stopping AI speech due to interruption...")
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return
                time.sleep(0.05)  # Check more frequently for interruptions
            
            if process.returncode == 0:
                print("✅ AI finished speaking")
                return
                
        except FileNotFoundError:
            continue
        except Exception as e:
            continue
    
    print("ERROR: No working audio player found.")

def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    """Robust non-streaming version with better error handling"""
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            stream=False
        )
        
        # More robust response parsing
        if response and hasattr(response, 'choices') and response.choices:
            if len(response.choices) > 0 and hasattr(response.choices[0], 'message'):
                full_response = response.choices[0].message.content
                if full_response:
                    print(NEON_GREEN + full_response + RESET_COLOR)
                    
                    # Log to file
                    with open(chat_log_filename, "a") as log_file:
                        log_file.write(f"User: {user_input}\n")
                        log_file.write(f"{bot_name}: {full_response}\n")
                    
                    return full_response
                else:
                    return "I received an empty response."
            else:
                return "I couldn't parse the response."
        else:
            return "I didn't receive a proper response."
            
    except Exception as e:
        print(f"ERROR in LM Studio API call: {e}")
        return "I'm having trouble connecting to the language model."

def transcribe_with_whisper(audio_file_path):
    """Transcribe audio using Whisper"""
    try:
        model = whisper.load_model("base.en")
        result = model.transcribe(audio_file_path)
        return result["text"].strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def is_likely_echo(transcribed_text, recent_ai_responses, similarity_threshold=0.6):
    """Check if transcribed text is likely an echo of AI's recent responses"""
    if not recent_ai_responses:
        return False
    
    transcribed_lower = transcribed_text.lower()
    
    for ai_response in recent_ai_responses:
        ai_lower = ai_response.lower()
        
        # Simple word overlap check
        transcribed_words = set(transcribed_lower.split())
        ai_words = set(ai_lower.split())
        
        if len(transcribed_words) > 0:
            overlap = len(transcribed_words.intersection(ai_words))
            similarity = overlap / len(transcribed_words)
            
            if similarity > similarity_threshold:
                return True
    
    return False

def audio_processor():
    """Process audio files from the queue"""
    conversation_history = []
    recent_ai_responses = deque(maxlen=3)  # Keep last 3 AI responses for echo detection
    
    # Check if chatbot1.txt exists
    system_file = "chatbot1.txt"
    if os.path.exists(system_file):
        system_message = open_file(system_file)
    else:
        print(f"⚠️  Warning: {system_file} not found, using default system message")
        system_message = "You are a helpful AI assistant named Julie."
    
    while not stop_listening.is_set():
        try:
            # Get audio file from queue with timeout
            audio_file = audio_queue.get(timeout=1)
            
            print("🔄 Transcribing...")
            user_input = transcribe_with_whisper(audio_file)
            
            # Clean up temporary file
            if os.path.exists(audio_file):
                os.remove(audio_file)
            
            if not user_input.strip():
                print("⚠️  No speech detected, continuing...")
                continue
            
            # Check for echo (AI hearing itself)
            if is_likely_echo(user_input, recent_ai_responses):
                print("🔇 Echo detected, ignoring...")
                continue
            
            if user_input.lower().strip() in ["exit", "quit", "stop", "goodbye"]:
                print("👋 Goodbye!")
                stop_listening.set()
                break
            
            print(CYAN + "You: " + user_input + RESET_COLOR)
            conversation_history.append({"role": "user", "content": user_input})
            
            print(PINK + "Julie:" + RESET_COLOR)
            
            # Reset stop speaking flag for new response
            stop_speaking.clear()
            
            chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Julie")
            conversation_history.append({"role": "assistant", "content": chatbot_response})
            
            # Store AI response for echo detection
            recent_ai_responses.append(chatbot_response)
            
            # Check if response is empty or just whitespace
            if not chatbot_response or len(chatbot_response.strip()) == 0:
                print("WARNING: Empty response from chatbot, skipping TTS")
                continue
            
            # Only proceed with TTS if we have a valid response and haven't been interrupted
            if not stop_speaking.is_set():
                try:
                    style = "default"
                    audio_file_pth = "/home/vm/Documents/Voice/voice/princess.wav"
                    
                    # Check if the voice file exists
                    if not os.path.exists(audio_file_pth):
                        print(f"⚠️  Voice file not found: {audio_file_pth}")
                        print("⚠️  Skipping voice conversion, using default TTS")
                        continue
                    
                    process_and_play_interruptible(chatbot_response, style, audio_file_pth)
                except Exception as e:
                    print(f"TTS Error: {e}")
                    continue
            
            # Keep conversation history manageable
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in audio processor: {e}")
            continue

def live_conversation():
    """Main function to start live conversation"""
    print("🚀 Starting Live Conversation Mode")
    print("💡 Tips:")
    print("   - Speak naturally and pause when done")
    print("   - You can interrupt the AI while it's speaking")
    print("   - The system will auto-calibrate to your environment")
    print("   - Say 'exit', 'quit', 'stop', or 'goodbye' to end")
    print("   - Press Ctrl+C to force quit")
    print("\n" + "="*50)
    
    # Start threads
    audio_capture_thread = threading.Thread(target=continuous_audio_capture, daemon=True)
    audio_processor_thread = threading.Thread(target=audio_processor, daemon=True)
    
    try:
        audio_capture_thread.start()
        audio_processor_thread.start()
        
        # Keep main thread alive
        while not stop_listening.is_set():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        stop_listening.set()
        stop_speaking.set()
        
    finally:
        # Wait for threads to finish
        if audio_capture_thread.is_alive():
            audio_capture_thread.join(timeout=2)
        if audio_processor_thread.is_alive():
            audio_processor_thread.join(timeout=2)
        
        print("✅ Live conversation ended")

if __name__ == "__main__":
    live_conversation()