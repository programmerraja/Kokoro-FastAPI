# type: ignore
import asyncio
import websockets
import numpy as np

# import tempfile
import os
import logging
import re
import time
import json
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple
import torch
from TTS.Kokoro.text_processing.text_processor import smart_split
from kokoro import KModel, KPipeline
from loguru import logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration parameters
class KokoroConfig:
    def __init__(
        self,
        voice: str = "af_bella",
        speed: float = 1.0,
        lang_code: str = "a",  # a=English, b=British, etc.
        volume_multiplier: float = 1.0,
    ):
        self.voice = voice
        self.speed = max(0.1, min(5.0, speed))
        self.lang_code = lang_code
        self.volume_multiplier = max(0.1, min(5.0, volume_multiplier))


# Optimize device selection
def get_optimal_device():
    """Determine the best device for the system"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


device = get_optimal_device()
logger.info(f"Using device: {device}")

# Global model and pipeline
model: Optional[KModel] = None
pipelines: Dict[str, KPipeline] = {}
temp_files: List[str] = []

logger.info("Loading Kokoro model...")

# Model and voice paths
MODEL_DIR = "api/src/models/v1_0"
VOICES_DIR = "api/src/voices/v1_0"
MODEL_FILE = "kokoro-v1_0.pth"
CONFIG_FILE = "config.json"


def cleanup_temp_files():
    """Clean up temporary files"""
    global temp_files
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except OSError as e:
            logger.warning(f"Error cleaning up temp file {temp_file}: {e}")
    temp_files.clear()


def get_pipeline(lang_code: str) -> KPipeline:
    """Get or create pipeline for language code"""
    global model, pipelines

    if not model:
        raise RuntimeError("Model not loaded")

    if lang_code not in pipelines:
        logger.info(f"Creating new pipeline for language code: {lang_code}")
        pipelines[lang_code] = KPipeline(
            lang_code=lang_code, model=model, device=device
        )
    return pipelines[lang_code]


async def load_model():
    """Load the Kokoro model"""
    global model

    try:
        # Construct paths
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        config_path = os.path.join(MODEL_DIR, CONFIG_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading Kokoro model on {device}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Config path: {config_path}")

        # Load model
        model = KModel(config=config_path, model=model_path).eval()

        # Move to appropriate device
        if device == "mps":
            logger.info("Moving model to MPS device")
            model = model.to(torch.device("mps"))
        elif device == "cuda":
            model = model.cuda()
        else:
            model = model.cpu()

        logger.info("Kokoro model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


async def get_voice_path(voice_name: str) -> str:
    """Get path to voice file"""
    voice_path = os.path.join(VOICES_DIR, f"{voice_name}.pt")
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice file not found: {voice_path}")
    return voice_path


async def run_tts(
    text: str, tts_config: KokoroConfig = None
) -> AsyncGenerator[bytes, None]:
    """Generate speech from text using Kokoro"""
    global model

    if not model:
        logger.error("Kokoro model not available")
        yield b"Error: Kokoro model not available"
        return

    if not tts_config:
        tts_config = KokoroConfig()

    logger.debug(f"Generating TTS for: [{text}]")

    try:
        # Get voice path
        voice_path = await get_voice_path(tts_config.voice)

        # Get pipeline for language
        pipeline = get_pipeline(tts_config.lang_code)

        # Generate audio
        logger.debug(
            f"Generating audio with voice: {tts_config.voice}, lang: {tts_config.lang_code}"
        )
        async for chunk_text in smart_split(
            text, tts_config.lang_code
        ):
            if chunk_text.strip():
                for result in pipeline(
                    chunk_text, voice=voice_path, speed=tts_config.speed, model=model
                ):
                    if result.audio is not None:
                        # Apply volume multiplier
                        audio_data = result.audio * tts_config.volume_multiplier

                        # Convert to int16 PCM bytes
                        audio_bytes = (
                            (audio_data.cpu().numpy() * 32767)
                            .astype(np.int16)
                            .tobytes()
                        )
                        yield audio_bytes
                    else:
                        logger.warning("No audio in chunk")

    except Exception as e:
        logger.error(f"TTS generation error: {e}")
        yield f"ERROR: {str(e)}".encode()
    finally:
        cleanup_temp_files()


async def handler(websocket):
    logger.info("New WebSocket connection established")

    try:
        async for message in websocket:
            start_time = time.perf_counter()

            try:
                # Parse message
                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                # Try to parse as JSON for config, fallback to plain text
                try:
                    data = json.loads(message)
                    text = data.get("text", "")
                    config_data = data.get("config", {})
                    tts_config = KokoroConfig(**config_data)
                except (json.JSONDecodeError, TypeError):
                    # Fallback to plain text
                    text = message
                    tts_config = KokoroConfig()

                if not text:
                    await websocket.send("ERROR: No text provided".encode())
                    continue

                # Generate and send audio
                async for audio_chunk in run_tts(text, tts_config):
                    await websocket.send(audio_chunk)

                processing_time = time.perf_counter() - start_time
                logger.info(f"Total processing time: {processing_time:.3f} seconds")

            except Exception as e:
                logger.error(f"Handler error: {e}")
                await websocket.send(f"ERROR: {str(e)}".encode())

    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        cleanup_temp_files()


async def main():
    # Load the model
    await load_model()

    # Pre-warm the model with dummy text for faster first generation
    logger.info("Pre-warming model...")
    try:
        dummy_text = "Hello, this is a test."
        async for _ in run_tts(dummy_text):
            pass
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Model pre-warming failed: {e}")

    # List available voices
    try:
        voices = [
            f.replace(".pt", "") for f in os.listdir(VOICES_DIR) if f.endswith(".pt")
        ]
        logger.info(f"Available voices: {voices}")
    except Exception as e:
        logger.warning(f"Could not list voices: {e}")

    async with websockets.serve(handler, "0.0.0.0", 9801):
        logger.info("Kokoro WebSocket server listening on ws://0.0.0.0:9801")
        logger.info(f"Model device: {device}")
        logger.info("Send text or JSON with 'text' and 'config' fields")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
