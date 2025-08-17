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

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="not-needed")

# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

# Function to play audio using PyAudio
def play_audio(file_path):
    """
    Use system audio player - much more reliable than PyAudio
    """
    import subprocess
    import os
    
    if not os.path.exists(file_path):
        print(f"ERROR: Audio file not found: {file_path}")
        return
    
    print(f"DEBUG - Playing audio file: {file_path}")
    
    # Try different system audio players
    audio_players = [
        ['aplay', file_path],
        ['paplay', file_path],
        ['ffplay', '-nodisp', '-autoexit', file_path],
        ['mpg123', file_path],
        ['cvlc', '--intf', 'dummy', '--play-and-exit', file_path]
    ]
    
    for player_cmd in audio_players:
        try:
            print(f"DEBUG - Trying player: {player_cmd[0]}")
            result = subprocess.run(
                player_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30  # 30 second timeout
            )
            if result.returncode == 0:
                print(f"DEBUG - Successfully played with {player_cmd[0]}")
                return
            else:
                print(f"DEBUG - {player_cmd[0]} failed with code {result.returncode}")
        except FileNotFoundError:
            print(f"DEBUG - {player_cmd[0]} not found")
            continue
        except subprocess.TimeoutExpired:
            print(f"DEBUG - {player_cmd[0]} timed out")
            continue
        except Exception as e:
            print(f"DEBUG - {player_cmd[0]} error: {e}")
            continue
    
    print("ERROR: No working audio player found. Install aplay, paplay, ffplay, mpg123, or vlc")
    print("Try: sudo apt install alsa-utils pulseaudio-utils ffmpeg mpg123 vlc")

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
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_pth):
    import os
    
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se

    speaker_wav = audio_file_pth

    print(f"DEBUG - Reference audio file: {audio_file_pth}")
    print(f"DEBUG - File exists: {os.path.exists(audio_file_pth)}")
    
    if os.path.exists(audio_file_pth):
        file_size = os.path.getsize(audio_file_pth)
        print(f"DEBUG - Reference file size: {file_size} bytes")
    else:
        print("ERROR - Reference audio file not found!")
        return

    # Process text and generate audio
    try:
        print("DEBUG - Starting voice extraction...")
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
        print(f"DEBUG - Voice extraction successful: {audio_name}")

        src_path = f'{output_dir}/tmp.wav'
        print(f"DEBUG - Generating TTS to: {src_path}")
        tts_model.tts(prompt, src_path, speaker=style, language='English')
        print(f"DEBUG - TTS generation complete")

        save_path = f'{output_dir}/output.wav'
        print(f"DEBUG - Starting voice conversion to: {save_path}")
        
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path, 
            message=encode_message
        )
        print(f"DEBUG - Voice conversion complete")

        print("Audio generated successfully.")
        
        # Play the voice-converted version (not the original TTS)
        print(f"DEBUG - Playing converted audio: {save_path}")
        play_audio(save_path)  # Use the converted audio, not src_path!

    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: play original TTS if voice conversion fails
        if os.path.exists(f'{output_dir}/tmp.wav'):
            print("Playing fallback TTS audio...")
            play_audio(f'{output_dir}/tmp.wav')


def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    """
    Robust non-streaming version with better error handling
    """
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    
    print(f"DEBUG - Sending to LM Studio:")
    print(f"DEBUG - User input: '{user_input}'")
    
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=messages,
            stream=False
        )
        
        print(f"DEBUG - Full response object: {response}")
        
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
                    
                    print(f"DEBUG - Response: '{full_response}'")
                    print(f"DEBUG - Response length: {len(full_response)}")
                    return full_response
                else:
                    print("DEBUG - Response content is None")
                    return "I received an empty response."
            else:
                print("DEBUG - No message in choices[0]")
                return "I couldn't parse the response."
        else:
            print("DEBUG - No choices in response")
            return "I didn't receive a proper response."
        
    except Exception as e:
        print(f"ERROR in LM Studio API call: {e}")
        print(f"ERROR type: {type(e)}")
        import traceback
        traceback.print_exc()
        return "I'm having trouble connecting to the language model."

def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model("base.en")  # You can choose different model sizes like 'tiny', 'base', 'small', 'medium', 'large'

    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]

# Function to record audio from the microphone and save to a file
def record_audio(filename):
    p = pyaudio.PyAudio()
    
    # Increase buffer size and add exception handling
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=44100,  # or whatever rate is working now
        input=True,
        frames_per_buffer=4096  # Increased from 1024
    )
    
    print("Recording...")
    frames = []
    
    try:
        for i in range(0, int(44100 / 4096 * 5)):  # Record for 5 seconds
            try:
                data = stream.read(4096, exception_on_overflow=False)  # Add this flag
                frames.append(data)
            except OSError as e:
                if e.errno == -9981:  # Input overflow
                    print("Buffer overflow, continuing...")
                    continue
                else:
                    raise
    except KeyboardInterrupt:
        print("Recording stopped by user")
    
    print("Recording finished")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recording
    import wave
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

# New function to handle a conversation with a user
# Modified section of your user_chatbot_conversation function
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot1.txt")
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break

        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Julie:" + RESET_COLOR)
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        
        # ADD THIS DEBUGGING CODE HERE:
        print(f"DEBUG - Chatbot response: '{chatbot_response}'")
        print(f"DEBUG - Response length: {len(chatbot_response) if chatbot_response else 0}")
        print(f"DEBUG - Response stripped length: {len(chatbot_response.strip()) if chatbot_response else 0}")

        # Check if response is empty or just whitespace
        if not chatbot_response or len(chatbot_response.strip()) == 0:
            print("WARNING: Empty response from chatbot, skipping TTS")
            continue  # Skip to next recording
        
        # Only proceed with TTS if we have a valid response
        try:
            prompt2 = chatbot_response
            style = "default"
            audio_file_pth2 = "/home/vm/Documents/Voice/voice/princess.wav"
            print("DEBUG - Attempting TTS generation...")
            process_and_play(prompt2, style, audio_file_pth2)
        except Exception as e:
            print(f"TTS Error details: {e}")
            print(f"Failed with response: '{chatbot_response}'")
            print("Continuing to next recording...")
            continue

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

user_chatbot_conversation()  # Start the conversation
