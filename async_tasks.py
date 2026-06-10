# fmt: off
import os
import base64
import asyncio
from httpx import AsyncClient, Timeout
from collections import deque
from colorama import init, Fore, Back, Style
from dotenv import load_dotenv
from datetime import datetime
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
# fmt: on

load_dotenv()
init(autoreset=True)

# --- TTS provider selection -------------------------------------------------
# 'elevenlabs' (default, unchanged behavior) or '60db'. Set in .env.
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'elevenlabs').strip().lower()

# Shared expressiveness settings, expressed on the ElevenLabs 0.0-1.0 scale.
# They are converted to 60dB's 0-100 scale when that provider is selected.
VOICE_STABILITY = 0.5
VOICE_SIMILARITY = 0.75

# --- ElevenLabs config ---
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID', 'ZucSKjgS1P6bwfYYa3I4')
ELEVENLABS_MODEL_ID = os.getenv('ELEVENLABS_MODEL_ID', 'eleven_turbo_v2')
# eleven_multilingual_v2 is the higher quality / slower alternative

# --- 60dB config ---
SIXTYDB_API_KEY = os.getenv('SIXTYDB_API_KEY')
SIXTYDB_VOICE_ID = os.getenv('SIXTYDB_VOICE_ID')  # UUID from GET /myvoices
SIXTYDB_OUTPUT_FORMAT = os.getenv('SIXTYDB_OUTPUT_FORMAT', 'mp3')  # mp3/wav/ogg/flac

print(Fore.YELLOW + f"TTS provider: {TTS_PROVIDER}" + Style.RESET_ALL)

file_increment = 0
audio_queue = deque()

# Define the text_to_speech_queue as a global asyncio Queue that can be accessed by main script
text_to_speech_queue = asyncio.Queue()

shutdown_event = asyncio.Event()

directory = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs(f'output/{directory}', exist_ok=True)


async def _synthesize_elevenlabs(client, text):
    """Call ElevenLabs REST TTS. Returns raw audio bytes, or None on failure."""
    if ELEVENLABS_API_KEY is None:
        print(Fore.RED + "Error: ELEVENLABS_API_KEY is not set.")
        return None

    url = f'https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}'
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "model_id": ELEVENLABS_MODEL_ID,
        "text": text,
        "voice_settings": {"similarity_boost": VOICE_SIMILARITY, "stability": VOICE_STABILITY}
    }

    response = await client.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.content  # raw MP3 bytes

    print(f"Error generating speech (ElevenLabs): {response.status_code} - {response.text}")
    return None


async def _synthesize_60db(client, text):
    """Call 60dB REST TTS (/tts-synthesize). Returns raw audio bytes, or None on failure."""
    if SIXTYDB_API_KEY is None:
        print(Fore.RED + "Error: SIXTYDB_API_KEY is not set.")
        return None

    url = 'https://api.60db.ai/tts-synthesize'
    headers = {
        "Authorization": f"Bearer {SIXTYDB_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "stability": int(VOICE_STABILITY * 100),    # 0.0-1.0 -> 0-100
        "similarity": int(VOICE_SIMILARITY * 100),   # 0.0-1.0 -> 0-100
        "output_format": SIXTYDB_OUTPUT_FORMAT
    }
    if SIXTYDB_VOICE_ID:  # omit to let 60dB use its system default voice
        data["voice_id"] = SIXTYDB_VOICE_ID

    response = await client.post(url, json=data, headers=headers)
    if response.status_code != 200:
        print(f"Error generating speech (60dB): {response.status_code} - {response.text}")
        return None

    payload = response.json()
    if not payload.get("success", True):
        print(f"Error generating speech (60dB): {payload.get('message')}")
        return None

    audio_b64 = payload.get("audio_base64")
    if not audio_b64:
        print(f"Error generating speech (60dB): response contained no audio_base64")
        return None

    return base64.b64decode(audio_b64)  # 60dB returns base64, decode to raw bytes


async def process_text_to_speech(text):
    global file_increment
    # 60dB can emit non-mp3 containers; pygame can still play wav/ogg/flac.
    ext = SIXTYDB_OUTPUT_FORMAT if TTS_PROVIDER == '60db' else 'mp3'
    filename = f'output/{directory}/{file_increment}.{ext}'
    file_increment += 1

    timeout = Timeout(30.0, connect=60.0)
    async with AsyncClient(timeout=timeout) as client:
        if TTS_PROVIDER == '60db':
            audio_bytes = await _synthesize_60db(client, text)
        else:
            audio_bytes = await _synthesize_elevenlabs(client, text)

    # Same downstream contract regardless of provider: write a file, queue it.
    if audio_bytes:
        with open(filename, 'wb') as audio_file:
            audio_file.write(audio_bytes)
        audio_queue.append(filename)


async def play_audio():
    while True:
        if not pygame.mixer.music.get_busy() and audio_queue:
            pygame.mixer.music.load(audio_queue.popleft())
            pygame.mixer.music.play()
        await asyncio.sleep(0.1)


async def text_to_speech_consumer(text_to_speech_queue):
    while True:
        text = await text_to_speech_queue.get()
        await process_text_to_speech(text)
        text_to_speech_queue.task_done()


async def start_async_tasks(text_to_speech_queue):
    """Starts asynchronous tasks without directly calling loop.run_forever()."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    consumer_task = loop.create_task(
        text_to_speech_consumer(text_to_speech_queue))
    play_task = loop.create_task(play_audio())
    return consumer_task, play_task


async def stop_async_tasks():
    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]

    # Gather all tasks to let them finish with cancellation
    await asyncio.gather(*tasks, return_exceptions=True)
