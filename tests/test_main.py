import pytest
import os
import shutil
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock
import pytest_asyncio

# Import the FastAPI app instance from main.py
# We need to make sure the main script can be imported without starting the server
# by ensuring uvicorn.run is under `if __name__ == '__main__':` block in main.py
from main import app, UPLOAD_DIR, TRANSCRIPT_DIR

# Test Client Fixture
@pytest_asyncio.fixture(scope="function")
async def client():
    # Ensure test directories are clean before each test function
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(TRANSCRIPT_DIR):
        shutil.rmtree(TRANSCRIPT_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    
    # Use httpx.AsyncClient with ASGITransport pointing to the app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client_instance:
        yield client_instance

    # Cleanup after test function completes
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    if os.path.exists(TRANSCRIPT_DIR):
        shutil.rmtree(TRANSCRIPT_DIR)

# --- Test Cases ---

@pytest.mark.asyncio
async def test_get_root(client: AsyncClient):
    """Test the root endpoint serves HTML."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>German WAV Transcriber</title>" in response.text

@pytest.mark.asyncio
@patch('main.model') # Mock the loaded whisper model object in main.py
async def test_transcribe_success(mock_whisper_model, client: AsyncClient):
    """Test successful transcription of a dummy WAV file."""
    # Configure the mock model's transcribe method
    mock_transcribe_result = {"text": "Dies ist ein Test."}
    mock_whisper_model.transcribe.return_value = mock_transcribe_result

    # Create a dummy wav file content (doesn't need to be real audio for this test)
    dummy_wav_content = b'RIFFdataWAVEfmt '
    files = {'wav_file': ('test.wav', dummy_wav_content, 'audio/wav')}

    response = await client.post("/transcribe/", files=files)

    # Assertions
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "File transcribed successfully"
    assert "transcript_filename" in json_response
    transcript_filename = json_response["transcript_filename"]
    assert transcript_filename.endswith(".txt")

    # Check if transcript file was created and contains correct text
    transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)
    assert os.path.exists(transcript_path)
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert content == mock_transcribe_result["text"]

    # Check if mock transcribe was called correctly
    assert mock_whisper_model.transcribe.call_count == 1
    call_args, call_kwargs = mock_whisper_model.transcribe.call_args
    assert call_args[0].endswith(".wav") # Check path arg
    assert call_kwargs.get('language') == 'german' # Check language kwarg (based on default .env)

    # Check if temporary upload file was deleted
    # The first argument to transcribe is the temp path
    temp_upload_path = call_args[0]
    assert not os.path.exists(temp_upload_path)
    assert UPLOAD_DIR in temp_upload_path # Ensure it was in the right folder

@pytest.mark.asyncio
async def test_transcribe_invalid_file_type(client: AsyncClient):
    """Test uploading a non-WAV file (e.g., a text file)."""
    # Note: Current main.py logs a warning but allows processing non .wav / incorrect mime types.
    # If strict validation is added (HTTP 400), this test needs changing.
    
    dummy_txt_content = b'this is not audio'
    files = {'wav_file': ('test.txt', dummy_txt_content, 'text/plain')}

    # If the model is mocked, it might still "succeed".
    # If testing without mock, Whisper would likely raise an error here.
    # Let's assume for now the check happens before transcription
    # Since main.py currently only warns, we expect it to proceed and potentially fail at whisper
    # To properly test the *intended* rejection, we'd need to uncomment the HTTPException 
    # in main.py's validation checks.
    
    # For now, just assert it accepts the request (status 200 or 500 depending on mocked behavior)
    # We'll mock transcribe to avoid actual whisper errors for this API test
    with patch('main.model') as mock_whisper_model:
        mock_whisper_model.transcribe.return_value = {"text": "mocked text"}
        response = await client.post("/transcribe/", files=files)
    
    # Because main.py currently only *warns* but doesn't raise 400, it will try to transcribe.
    # Since we mocked it, it will return 200. If strict validation is added later, 
    # this should assert status_code == 400
    assert response.status_code == 200 
    # assert response.status_code == 400 
    # assert "Invalid file" in response.json().get("detail", "")

@pytest.mark.asyncio
@patch('main.model') # Mock the loaded whisper model object
async def test_transcribe_whisper_error(mock_whisper_model, client: AsyncClient):
    """Test handling of an error during the Whisper transcription process."""
    # Configure the mock model's transcribe method to raise an exception
    mock_whisper_model.transcribe.side_effect = Exception("Whisper failed!")

    dummy_wav_content = b'RIFFdataWAVEfmt '
    files = {'wav_file': ('test_error.wav', dummy_wav_content, 'audio/wav')}

    response = await client.post("/transcribe/", files=files)

    # Assertions
    assert response.status_code == 500 # Internal Server Error expected
    json_response = response.json()
    assert "Transcription failed" in json_response.get("detail", "")
    assert "Whisper failed!" in json_response.get("detail", "")

    # Check if temporary upload file was still deleted even after error
    # Need to know the temp path. Since transcribe failed, we can't get it from call_args.
    # We need to check the uploads folder is empty.
    assert not os.listdir(UPLOAD_DIR)

# TODO: Add tests for:
# - File saving errors (upload dir not writable?)
# - Transcript saving errors (transcript dir not writable?)
# - Very large file uploads (if implementing specific handling)
# - Cleanup failures (though harder to test reliably) 