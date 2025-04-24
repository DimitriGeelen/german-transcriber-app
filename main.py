import os
import uuid
import datetime
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import whisper
import logging
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv() # Load variables from .env file

# Get config from environment variables with defaults
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
SUPPORTED_LANGUAGE = os.getenv("SUPPORTED_LANGUAGE", "german")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
TRANSCRIPT_DIR = os.getenv("TRANSCRIPT_DIR", "./transcripts")
# Simple security check: prevent directory traversal
if ".." in UPLOAD_DIR or ".." in TRANSCRIPT_DIR:
    raise ValueError("UPLOAD_DIR and TRANSCRIPT_DIR cannot contain '..'")


# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialization ---
app = FastAPI()

# Load Whisper model - Log progress
logger.info(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
try:
    # Note: Whisper downloads models automatically on first use if not found
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    logger.info("Whisper model loaded successfully.")
    # Check language support (optional, Whisper often handles it)
    if SUPPORTED_LANGUAGE not in whisper.tokenizer.LANGUAGES:
         logger.warning(f"Whisper may not officially support language code '{SUPPORTED_LANGUAGE}', but attempting anyway.")

except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
    # Exit if model fails to load? Or handle gracefully? For now, log and continue, endpoints might fail.
    model = None # Indicate model failed to load

# Ensure directories exist
try:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    logger.info(f"Ensured directories exist: {UPLOAD_DIR}, {TRANSCRIPT_DIR}")
except OSError as e:
     logger.error(f"Error creating directories: {e}", exc_info=True)
     # Consider exiting if directories can't be created

# Mount static files (like CSS/JS if added later) - not strictly needed for index.html rendering via response
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper function to format seconds into HH:MM:SS.ms
def format_timestamp(seconds: float):
    """Formats seconds into HH:MM:SS.ms string."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

# --- Helper: Check for ffmpeg ---
def check_ffmpeg():
    """Checks if ffmpeg command is available in PATH."""
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found. Please install ffmpeg and ensure it is in your PATH.")
        logger.error("Whisper transcription will fail without ffmpeg.")
        return False
    logger.info("ffmpeg found in PATH.")
    return True

ffmpeg_available = check_ffmpeg() # Check on startup

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_root():
    """Serves the main HTML page."""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("templates/index.html not found")
        raise HTTPException(status_code=500, detail="Frontend file missing.")
    except Exception as e:
        logger.error(f"Error reading index.html: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not load frontend.")


@app.post("/transcribe/")
async def transcribe_audio(wav_file: UploadFile = File(...)):
    """Handles WAV file upload, transcription, and saving."""
    # Add check here before processing
    if not ffmpeg_available:
        raise HTTPException(status_code=503, detail="Server configuration error: ffmpeg is missing.")
        
    if not model:
        logger.error("Transcription endpoint called, but Whisper model failed to load.")
        raise HTTPException(status_code=503, detail="Transcription service unavailable.")

    # Basic validation: Check file extension (optional but good practice)
    if not wav_file.filename.lower().endswith(".wav"):
         logger.warning(f"Received file '{wav_file.filename}' which does not end with .wav")
         # Allow processing anyway, but log it. Could raise HTTPException here.
         # raise HTTPException(status_code=400, detail="Invalid file type. Only .wav files are accepted.")

    # Check content type (more reliable)
    if wav_file.content_type not in ["audio/wav", "audio/x-wav", "audio/wave"]:
         logger.warning(f"Received file '{wav_file.filename}' with unexpected content type: {wav_file.content_type}")
         # Allow processing, but log it. Could raise HTTPException here.
         # raise HTTPException(status_code=400, detail="Invalid file content type. Only WAV audio is accepted.")


    # Generate unique filenames
    unique_id = uuid.uuid4()
    temp_save_filename = f"{unique_id}.wav"
    temp_save_path = os.path.join(UPLOAD_DIR, temp_save_filename)
    transcript_filename = f"{unique_id}.txt"
    transcript_save_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)

    # 1. Save Uploaded File
    try:
        logger.info(f"Saving uploaded file to temporary path: {temp_save_path}")
        with open(temp_save_path, "wb") as buffer:
            shutil.copyfileobj(wav_file.file, buffer)
        logger.info(f"File saved successfully: {temp_save_path}")
    except Exception as e:
        logger.error(f"Error saving uploaded file {temp_save_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
    finally:
        # Ensure UploadFile's file pointer is closed
        await wav_file.close()


    transcription_result = None
    # 2. Transcribe Audio
    try:
        logger.info(f"Starting transcription for {temp_save_path} (Lang: {SUPPORTED_LANGUAGE})")
        # Call transcribe. The result dictionary contains segment data with timestamps.
        # verbose=True might provide more details if needed, but segments are default.
        # word_timestamps=True can be added for word-level detail, but increases processing.
        result = model.transcribe(temp_save_path, language=SUPPORTED_LANGUAGE)
        transcription_result = result # Store the full result
        logger.info(f"Transcription successful for {temp_save_path}")

    except Exception as e:
        logger.error(f"Error during transcription for {temp_save_path}: {e}", exc_info=True)
        # Clean up the saved file even if transcription fails
        try:
            os.remove(temp_save_path)
            logger.info(f"Cleaned up temporary file after transcription error: {temp_save_path}")
        except OSError as rm_err:
            logger.error(f"Error removing temporary file {temp_save_path} after transcription error: {rm_err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        # 3. Clean up uploaded file (always attempt this)
        if os.path.exists(temp_save_path):
            try:
                os.remove(temp_save_path)
                logger.info(f"Cleaned up temporary file: {temp_save_path}")
            except OSError as e:
                # Log error but don't fail the request if cleanup fails
                logger.error(f"Error removing temporary file {temp_save_path}: {e}", exc_info=True)


    # 4. Save Transcription with Timestamps
    try:
        logger.info(f"Saving transcription with timestamps to: {transcript_save_path}")
        with open(transcript_save_path, "w", encoding="utf-8") as f:
            if transcription_result and "segments" in transcription_result:
                for segment in transcription_result["segments"]:
                    start_time = format_timestamp(segment['start'])
                    end_time = format_timestamp(segment['end'])
                    text = segment['text'].strip()
                    f.write(f"[{start_time} --> {end_time}] {text}\n")
            elif transcription_result and "text" in transcription_result:
                # Fallback if segments aren't available for some reason
                logger.warning("Transcription result missing 'segments', saving full text only.")
                f.write(transcription_result["text"])
            else:
                 logger.error("Transcription result object was empty or invalid.")
                 # Write an empty file or indicate error?
                 f.write("[Error: Transcription failed to produce valid result]")

        logger.info(f"Transcription saved successfully: {transcript_save_path}")
    except Exception as e:
        logger.error(f"Error saving transcript file {transcript_save_path}: {e}", exc_info=True)
        # Log error, but the transcription was successful, so maybe still return success?
        # Or raise 500? Let's raise 500 as saving the result is critical.
        raise HTTPException(status_code=500, detail=f"Could not save transcript file: {e}")

    # 5. Return Response
    logger.info(f"Request completed successfully. Transcript filename: {transcript_filename}")
    return {
        "message": "File transcribed successfully (with timestamps)",
        "transcript_filename": transcript_filename,
    }

# --- Run Server (for local development) ---
if __name__ == "__main__":
    # Check ffmpeg again just before running server locally
    if not ffmpeg_available:
        logger.critical("Cannot start server: ffmpeg is not installed or not in PATH.")
        exit(1) # Exit if ffmpeg is missing for local run
        
    import uvicorn
    logger.info("Starting Uvicorn server for development...")
    # Allow specifying host and port via env vars too, e.g., HOST=0.0.0.0 PORT=8001
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port) 