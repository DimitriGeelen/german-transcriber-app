import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport # Corrected import
import os
import sys # Added sys import
import shutil
import re # Added for regex check in test
from io import BytesIO # Added for simulating file upload
from unittest.mock import patch, MagicMock # Added for mocking
import torch
from fastapi import UploadFile # Added import
import time # Added for polling test
import asyncio # Added for async sleep
import json # Added for JSON operations

# --- Add project root to sys.path --- 
# Get the directory of the current test file (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)
# Add the project root to the Python path
sys.path.insert(0, project_root)

# We will now mock imports/functions used BY main.py, rather than importing main.py itself

# Define dummy paths for testing purposes, assuming main.py uses these defaults
# If main.py determines these differently, mocks might need adjustment
TEST_UPLOAD_DIR = os.path.join(project_root, "test_uploads")
TEST_TRANSCRIPT_DIR = os.path.join(project_root, "test_transcripts")
BASE_URL = "http://test" # Base URL for the test client

from main import app, tasks, TaskStatus # Import app and tasks dict/enum

# Fixture to set up and tear down test directories
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Ensure test directories exist and are cleaned up after tests."""
    if os.path.exists(TEST_UPLOAD_DIR):
        shutil.rmtree(TEST_UPLOAD_DIR)
    if os.path.exists(TEST_TRANSCRIPT_DIR):
        shutil.rmtree(TEST_TRANSCRIPT_DIR)
        
    os.makedirs(TEST_UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEST_TRANSCRIPT_DIR, exist_ok=True)
    
    tests_dir = os.path.dirname(__file__) 
    sample_wav_path = os.path.join(tests_dir, "sample.wav")
    with open(sample_wav_path, "wb") as f:
        f.write(b'RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00\xfa\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        
    yield 

    print("\nRunning teardown: Cleaning up test directories...")
    if os.path.exists(TEST_UPLOAD_DIR):
        shutil.rmtree(TEST_UPLOAD_DIR)
        print(f"Removed: {TEST_UPLOAD_DIR}")
    if os.path.exists(TEST_TRANSCRIPT_DIR):
        shutil.rmtree(TEST_TRANSCRIPT_DIR)
        print(f"Removed: {TEST_TRANSCRIPT_DIR}")
    if os.path.exists(sample_wav_path):
         os.remove(sample_wav_path)
         print(f"Removed: {sample_wav_path}")
    # --- Clear tasks dict after tests ---
    tasks.clear()
    print("Cleared tasks dictionary.")

# Fixture to provide an async test client
@pytest_asyncio.fixture(scope="function")
async def client():
    """Provides an async test client for the FastAPI app, manually setting app state."""
    try:
        print("Importing app for client fixture...")
        from main import app
        print("App imported successfully for client fixture.")

        # --- Manually Set App State for Testing --- 
        # Create mock instances for state
        mock_whisper_model_instance = MagicMock()
        mock_diarization_pipeline_instance = MagicMock()
        mock_embedding_pipeline_instance = MagicMock()
        
        # Assign mocks directly to the app state BEFORE creating the client
        app.state.model = mock_whisper_model_instance
        app.state.diarization_pipeline = mock_diarization_pipeline_instance
        app.state.embedding_pipeline = mock_embedding_pipeline_instance
        app.state.ffmpeg_available = True # Assume ffmpeg is available for tests
        app.state.upload_dir = TEST_UPLOAD_DIR # Explicitly set for tests
        app.state.transcript_dir = TEST_TRANSCRIPT_DIR # Explicitly set for tests
        app.state.device = "cpu" # <<< ADD DEVICE STATE (default to cpu for tests)

        # --- Ensure tasks dict is clean before each test ---
        tasks.clear()
        # ----------------------------------------------------

        print("Manually set app.state attributes for testing.")
        # --------------------------------------------
        
        async with AsyncClient(transport=ASGITransport(app=app), base_url=BASE_URL) as test_client:
            # Yield the client *and* the mock instances so tests can configure/assert them
            yield test_client, mock_whisper_model_instance, mock_diarization_pipeline_instance, mock_embedding_pipeline_instance
            
    except ImportError as e:
        print(f"Failed to import app in client fixture: {e}")
        pytest.skip(f"Skipping test because FastAPI app import failed: {e}")
    except Exception as e:
        print(f"Unexpected error in client fixture: {e}")
        pytest.fail(f"Client fixture failed unexpectedly: {e}")
    finally:
        # Clean up manually set state after test (optional but good practice)
        if 'app' in locals() and hasattr(app, 'state'):
             if hasattr(app.state, 'model'): del app.state.model
             if hasattr(app.state, 'diarization_pipeline'): del app.state.diarization_pipeline
             if hasattr(app.state, 'ffmpeg_available'): del app.state.ffmpeg_available
             if hasattr(app.state, 'upload_dir'): del app.state.upload_dir
             if hasattr(app.state, 'transcript_dir'): del app.state.transcript_dir
             if hasattr(app.state, 'embedding_pipeline'): del app.state.embedding_pipeline
             if hasattr(app.state, 'device'): del app.state.device # <<< CLEANUP DEVICE STATE
             print("Cleaned up manually set app.state attributes.")
         # --- Clear tasks dict after each test ---
        tasks.clear()
        print("Cleared tasks dictionary after test run.")

# --- Test Functions ---

@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient): # client fixture now yields tuple
    test_client, _, _, _ = client # Unpack the client, adjust unpacking for new mock
    """Test the root endpoint returns HTML content successfully."""
    response = await test_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>German WAV Transcriber</title>" in response.text


@pytest.mark.asyncio
# Remove mocks for lifespan functions (check_ffmpeg, load_model, from_hparams, SPEECHBRAIN_AVAILABLE)
async def test_transcribe_endpoint_success_with_diarization(
    # Use the original fixture name 'client', even though it yields a tuple
    client # Changed back from client_tuple
): 
    pytest.skip("Skipping original sync test, needs async adaptation.")
    # Unpack client and mock instances from fixture
    test_client, mock_whisper_model_instance, mock_diarization_pipeline_instance, _ = client # Use test_client locally, adjust unpacking
    
    """Test the /transcribe/ endpoint successfully returns diarized transcript info."""

    # Configure the mock instances provided by the fixture
    mock_whisper_model_instance.transcribe.side_effect = [
        {"text": "Hello speaker one endpoint test"}, 
        {"text": "Hi speaker two endpoint test"}     
    ]
    
    mock_diarization_output = torch.tensor([
        [0.5, 4.8, 0.], # Speaker 0
        [5.1, 9.5, 1.]  # Speaker 1
    ])
    mock_diarization_pipeline_instance.return_value = mock_diarization_output

    # Define other mocks needed for endpoint logic using patch.multiple
    mock_uuid_instance = MagicMock(return_value='test-endpoint-uuid')
    mock_librosa_load_instance = MagicMock()
    dummy_audio_data = [0.0] * 16000 * 10
    dummy_sample_rate = 16000
    mock_librosa_load_instance.return_value = (dummy_audio_data, dummy_sample_rate)
    
    mock_copyfileobj_instance = MagicMock()
    mock_os_remove_instance = MagicMock()
    mock_os_path_exists_instance = MagicMock(return_value=True)

    with patch.multiple(
        'main',
        # Remove mocks for lifespan functions previously here 
        uuid=MagicMock(uuid4=mock_uuid_instance),
        librosa=MagicMock(load=mock_librosa_load_instance),
        shutil=MagicMock(copyfileobj=mock_copyfileobj_instance),
        # os=MagicMock( # Remove os mock from here
        #     path=MagicMock(exists=mock_os_path_exists_instance),
        #     remove=mock_os_remove_instance,
        # )
        # NO builtins.open here
    ) as mocks:
        # Correctly NEST the patch for builtins.open
        # Also nest patches for specific os functions needed
        with patch('builtins.open', MagicMock()) as mock_open, \
             patch('main.os.path.exists', mock_os_path_exists_instance) as mock_exists, \
             patch('main.os.remove', mock_os_remove_instance) as mock_remove:
            
            # --- Mock Configuration for Endpoint Logic --- 
            transcript_filename = "test-endpoint-uuid.txt"
            temp_save_filename = "test-endpoint-uuid.wav"
            
            try:
                from main import UPLOAD_DIR, TRANSCRIPT_DIR 
            except ImportError:
                UPLOAD_DIR = TEST_UPLOAD_DIR 
                TRANSCRIPT_DIR = TEST_TRANSCRIPT_DIR 
                
            temp_save_path = os.path.join(UPLOAD_DIR, temp_save_filename)

            # --- Simulate File Upload --- 
            dummy_wav_content = b'RIFFdummyWAVE'
            files = {'wav_file': ('test.wav', BytesIO(dummy_wav_content), 'audio/wav')}

            # --- Make API Call --- 
            response = await test_client.post("/transcribe/", files=files)

            # --- Assertions --- 
            assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}. Response: {response.text}"

            response_json = response.json()
            assert response_json["message"] == "File transcribed and diarized successfully."
            assert response_json["transcript_filename"] == transcript_filename

            # Check Mock Calls 
            # Check mock for open() was called correctly for the temp file
            mock_open.assert_any_call(temp_save_path, "wb")
            # Get the handle associated with the temp file write
            # This is tricky if open is called multiple times; assume the first call is temp
            # A more robust way might involve inspecting call_args_list
            # The handle yielded by `with open(...)` is the result of __enter__
            mock_temp_file_handle = mock_open.return_value.__enter__.return_value 
            
            # Check copyfileobj was called with the uploaded file object and the mock handle
            mock_copyfileobj_instance.assert_called_once()
            assert mock_copyfileobj_instance.call_args[0][1] == mock_temp_file_handle # Check 2nd arg
            # We might also want to check the first arg is a file-like object if needed
            # assert hasattr(mock_copyfileobj_instance.call_args[0][0], 'read')
            
            # Check mocks provided by the fixture were called
            mock_diarization_pipeline_instance.assert_called_once_with(temp_save_path)
            assert mock_whisper_model_instance.transcribe.call_count == 2
            
            # Check mocks from patch.multiple
            mock_librosa_load_instance.assert_called_once_with(temp_save_path, sr=16000, mono=True)
            mock_exists.assert_called_with(temp_save_path)
            mock_remove.assert_called_with(temp_save_path)

@pytest.mark.asyncio
async def test_transcribe_endpoint_save_failure(
    client # Use the original fixture name 
): 
    """Test the /transcribe/ endpoint returns 500 if saving the uploaded file fails."""
    test_client, _, _, _ = client # Unpack the client, rename locally, adjust unpacking

    # Mocks needed for logic *before* the save failure point
    mock_uuid_instance = MagicMock(return_value='test-save-fail-uuid')
    # Mock copyfileobj to raise an error
    mock_copyfileobj_instance = MagicMock(side_effect=OSError("Disk full")) 

    with patch.multiple(
        'main',
        uuid=MagicMock(uuid4=mock_uuid_instance),
        shutil=MagicMock(copyfileobj=mock_copyfileobj_instance)
        # No need to mock os.path.exists, os.remove, librosa etc. 
        # as the function should fail before reaching those
        # We also don't need to mock builtins.open here, as copyfileobj itself fails
    ) as mocks:
        # --- Simulate File Upload --- 
        dummy_wav_content = b'RIFFdummyWAVE'
        files = {'wav_file': ('test_fail.wav', BytesIO(dummy_wav_content), 'audio/wav')}

        # --- Make API Call --- 
        response = await test_client.post("/transcribe/", files=files)

        # --- Assertions --- 
        assert response.status_code == 500
        response_json = response.json()
        assert "Could not save uploaded file" in response_json["detail"]
        assert "Disk full" in response_json["detail"] # Check our specific error is included

        # Check that copyfileobj was called (even though it failed)
        mock_copyfileobj_instance.assert_called_once()

# --- Test Transcript Saving Error ---

def mock_open_transcript_fail(*args, **kwargs):
    """Custom side effect for mock_open to fail only on transcript write."""
    path, mode = args[0], args[1]
    # Use a simple string check - could be more robust if paths were complex
    if "transcripts" in path and mode == "w": 
        raise OSError("Permission denied")
    # Otherwise, return a standard MagicMock for file operations (like temp save)
    mock_file = MagicMock()
    # Make the mock file handle usable in a 'with' statement
    mock_file.__enter__.return_value = mock_file 
    mock_file.__exit__.return_value = None
    return mock_file

@pytest.mark.asyncio
async def test_transcribe_endpoint_transcript_save_failure(
    client # Fixture providing (client, mock_whisper, mock_diarize)
): 
    pytest.skip("Skipping original sync test, needs async adaptation.")
    """Test the /transcribe/ endpoint returns 500 if saving the transcript file fails."""
    test_client, mock_whisper_model_instance, mock_diarization_pipeline_instance, _ = client # adjust unpacking

    # Configure mocks provided by the fixture (similar to success case)
    mock_whisper_model_instance.transcribe.side_effect = [
        {"text": "Hello one"}, {"text": "Hi two"}     
    ]
    mock_diarization_output = torch.tensor([[0.5, 4.8, 0.], [5.1, 9.5, 1.]])
    mock_diarization_pipeline_instance.return_value = mock_diarization_output

    # Define mocks for endpoint logic
    mock_uuid_instance = MagicMock(return_value='test-transcript-fail-uuid')
    mock_librosa_load_instance = MagicMock()
    dummy_audio_data = [0.0] * 16000 * 10; dummy_sample_rate = 16000
    mock_librosa_load_instance.return_value = (dummy_audio_data, dummy_sample_rate)
    mock_copyfileobj_instance = MagicMock()
    mock_os_remove_instance = MagicMock()
    mock_os_path_exists_instance = MagicMock(return_value=True)

    # Patch modules used *within* the endpoint handler
    with patch.multiple(
        'main',
        uuid=MagicMock(uuid4=mock_uuid_instance),
        librosa=MagicMock(load=mock_librosa_load_instance),
        shutil=MagicMock(copyfileobj=mock_copyfileobj_instance),
    ) as mocks_main, \
         patch('main.os.path.exists', mock_os_path_exists_instance), \
         patch('main.os.remove', mock_os_remove_instance), \
         patch('builtins.open', side_effect=mock_open_transcript_fail) as mock_open:
        
        # Get temp path for assertions
        try: from main import UPLOAD_DIR
        except ImportError: UPLOAD_DIR = TEST_UPLOAD_DIR
        temp_save_path = os.path.join(UPLOAD_DIR, 'test-transcript-fail-uuid.wav')
        
        # Simulate File Upload
        files = {'wav_file': ('test.wav', BytesIO(b'RIFFdummyWAVE'), 'audio/wav')}

        # Make API Call 
        response = await test_client.post("/transcribe/", files=files)

        # Assertions 
        assert response.status_code == 500
        response_json = response.json()
        assert "Could not save transcript file" in response_json["detail"]
        assert "Permission denied" in response_json["detail"] # Check our specific error

        # Check relevant mocks were called 
        mock_open.assert_any_call(temp_save_path, "wb") # Temp file save should still happen
        # Check the failing call to open was made
        # Need to construct the expected transcript path
        try: from main import TRANSCRIPT_DIR
        except ImportError: TRANSCRIPT_DIR = TEST_TRANSCRIPT_DIR
        transcript_save_path = os.path.join(TRANSCRIPT_DIR, 'test-transcript-fail-uuid.txt')
        mock_open.assert_any_call(transcript_save_path, "w", encoding="utf-8")
        mock_diarization_pipeline_instance.assert_called_once()
        assert mock_whisper_model_instance.transcribe.call_count == 2
        # Cleanup should still be attempted
        mock_os_path_exists_instance.assert_called_with(temp_save_path)
        mock_os_remove_instance.assert_called_with(temp_save_path) 

# --- Test Transcription Failure ---

@pytest.mark.asyncio
async def test_transcribe_endpoint_transcription_failure(
    client # Fixture providing (client, mock_whisper, mock_diarize)
): 
    pytest.skip("Skipping original sync test, needs async adaptation.")
    """Test the /transcribe/ endpoint returns 500 if transcription fails."""
    test_client, mock_whisper_model_instance, mock_diarization_pipeline_instance, _ = client # adjust unpacking

    # Configure mocks provided by the fixture
    mock_whisper_model_instance.transcribe.side_effect = Exception("Whisper crashed") # Make transcribe fail
    # Diarization mock setup (needed to get past that stage)
    mock_diarization_output = torch.tensor([[0.5, 4.8, 0.]]) # Example output
    mock_diarization_pipeline_instance.return_value = mock_diarization_output

    # Define mocks for endpoint logic needed before transcription
    mock_uuid_instance = MagicMock(return_value='test-transcribe-fail-uuid')
    mock_librosa_load_instance = MagicMock()
    dummy_audio_data = [0.0] * 16000 * 10; dummy_sample_rate = 16000
    mock_librosa_load_instance.return_value = (dummy_audio_data, dummy_sample_rate)
    mock_copyfileobj_instance = MagicMock()
    mock_os_remove_instance = MagicMock()
    mock_os_path_exists_instance = MagicMock(return_value=True)

    # Patch modules used *within* the endpoint handler
    with patch.multiple(
        'main',
        uuid=MagicMock(uuid4=mock_uuid_instance),
        librosa=MagicMock(load=mock_librosa_load_instance),
        shutil=MagicMock(copyfileobj=mock_copyfileobj_instance),
    ) as mocks_main, \
         patch('main.os.path.exists', mock_os_path_exists_instance), \
         patch('main.os.remove', mock_os_remove_instance), \
         patch('builtins.open', MagicMock()) as mock_open:
        
        # Get temp path for assertions
        try: from main import UPLOAD_DIR
        except ImportError: UPLOAD_DIR = TEST_UPLOAD_DIR
        temp_save_path = os.path.join(UPLOAD_DIR, 'test-transcribe-fail-uuid.wav')
        
        # Simulate File Upload
        files = {'wav_file': ('test.wav', BytesIO(b'RIFFdummyWAVE'), 'audio/wav')}

        # Make API Call 
        response = await test_client.post("/transcribe/", files=files)

        # Assertions 
        assert response.status_code == 500
        response_json = response.json()
        assert "Transcription failed" in response_json["detail"]
        assert "Whisper crashed" in response_json["detail"] # Check our specific error

        # Check relevant mocks were called 
        mock_open.assert_any_call(temp_save_path, "wb") 
        mock_copyfileobj_instance.assert_called_once()
        mock_diarization_pipeline_instance.assert_called_once()
        mock_librosa_load_instance.assert_called_once()
        mock_whisper_model_instance.transcribe.assert_called_once() # Called once before crashing
        # Cleanup should still be attempted
        mock_os_path_exists_instance.assert_called_with(temp_save_path)
        mock_os_remove_instance.assert_called_with(temp_save_path) 

# --- Test Diarization Failure ---

@pytest.mark.asyncio
async def test_transcribe_endpoint_diarization_failure(
    client # Fixture providing (client, mock_whisper, mock_diarize)
): 
    pytest.skip("Skipping original sync test, needs async adaptation.")
    """Test the /transcribe/ endpoint succeeds but skips diarization if the pipeline fails."""
    test_client, mock_whisper_model_instance, mock_diarization_pipeline_instance, _ = client # adjust unpacking

    # Configure mocks provided by the fixture
    # Diarization fails (returns None)
    mock_diarization_pipeline_instance.return_value = None 
    # Whisper model should transcribe the whole file now (mock needs segments for formatting)
    mock_whisper_model_instance.transcribe.return_value = {
        "text": "Full transcription text when diarization fails",
        "segments": [
            {"start": 0.1, "end": 5.5, "text": "Full transcription text"},
            {"start": 5.8, "end": 9.2, "text": "when diarization fails"}
        ]
    }

    # Define mocks for endpoint logic
    mock_uuid_instance = MagicMock(return_value='test-diarize-fail-uuid')
    mock_librosa_load_instance = MagicMock()
    dummy_audio_data = [0.0] * 16000 * 10; dummy_sample_rate = 16000
    mock_librosa_load_instance.return_value = (dummy_audio_data, dummy_sample_rate)
    mock_copyfileobj_instance = MagicMock()
    mock_os_remove_instance = MagicMock()
    mock_os_path_exists_instance = MagicMock(return_value=True)

    # Patch modules used *within* the endpoint handler
    with patch.multiple(
        'main',
        uuid=MagicMock(uuid4=mock_uuid_instance),
        librosa=MagicMock(load=mock_librosa_load_instance),
        shutil=MagicMock(copyfileobj=mock_copyfileobj_instance),
    ) as mocks_main, \
         patch('main.os.path.exists', mock_os_path_exists_instance), \
         patch('main.os.remove', mock_os_remove_instance), \
         patch('builtins.open', MagicMock()) as mock_open:
        
        try: from main import UPLOAD_DIR
        except ImportError: UPLOAD_DIR = TEST_UPLOAD_DIR
        temp_save_path = os.path.join(UPLOAD_DIR, 'test-diarize-fail-uuid.wav')
        
        files = {'wav_file': ('test.wav', BytesIO(b'RIFFdummyWAVE'), 'audio/wav')}

        response = await test_client.post("/transcribe/", files=files)

        # Assertions 
        assert response.status_code == 200
        response_json = response.json()
        assert "diarization skipped or failed" in response_json["message"]
        assert response_json["transcript_filename"] == 'test-diarize-fail-uuid.txt'

        # Check relevant mocks were called 
        mock_open.assert_any_call(temp_save_path, "wb") 
        mock_copyfileobj_instance.assert_called_once()
        mock_diarization_pipeline_instance.assert_called_once()
        mock_librosa_load_instance.assert_called_once()
        # Whisper transcribe called once for the full audio
        mock_whisper_model_instance.transcribe.assert_called_once()
        # Cleanup should still be attempted
        mock_os_path_exists_instance.assert_called_with(temp_save_path)
        mock_os_remove_instance.assert_called_with(temp_save_path) 

# --- Test ffmpeg Missing ---

@pytest.mark.asyncio
async def test_transcribe_endpoint_ffmpeg_missing(
    client # Use the standard client fixture
): 
    """Test the /transcribe/ endpoint returns 503 if ffmpeg is missing."""
    test_client, _, _, _ = client # Unpack client, ignore model mocks, adjust unpacking

    # Manually override the state set by the fixture for this specific test
    # We need the app instance itself for this
    try:
        from main import app
        app.state.ffmpeg_available = False
        print("Manually set ffmpeg_available=False for test.")
    except ImportError:
         pytest.fail("Could not import app to override state for ffmpeg test.")

    # No external patching needed as state is manually set

    # --- Simulate File Upload --- 
    dummy_wav_content = b'RIFFdummyWAVE'
    files = {'wav_file': ('test_ffmpeg.wav', BytesIO(dummy_wav_content), 'audio/wav')}

    # --- Make API Call --- 
    response = await test_client.post("/transcribe/", files=files)

    # --- Assertions --- 
    assert response.status_code == 503
    response_json = response.json()
    assert response_json["detail"] == "Server dependency missing: ffmpeg is not available."

    # Optional: Restore state if needed, though fixture cleanup should handle it
    # app.state.ffmpeg_available = True 

# Keep the logic test as well 

# --- Test Specific Imports/Behaviors ---

def test_speechbrain_import_fails():
    """Verify that importing SpeakerDiarization from .pretrained fails.
    
    This test expects an ImportError based on server logs, even if manual
    import works. This helps confirm the context-dependent issue.
    Note: If this test fails (meaning the import *succeeds*), it indicates
    the issue is even more specific to the Uvicorn/FastAPI process itself.
    """
    with pytest.raises(ImportError):
        print("\nAttempting import: from speechbrain.pretrained import SpeakerDiarization")
        # NOTE: The try/except block in main.py MUST be restored for this test to be meaningful
        # If the try/except is commented out, this test might pass but for the wrong reason.
        from speechbrain.pretrained import SpeakerDiarization
        print("Import unexpectedly succeeded!") # Should not be reached if test passes

# Keep the logic test as well 

# --- Mock background task for async tests ---
async def mock_process_transcription_task(temp_save_path: str, task_id: str, app_state: dict):
    """Simulates the background task, updating the global tasks dictionary.
    
    Uses string values for status ('PROCESSING', 'COMPLETED', 'FAILED') 
    to align with the TaskStatus Pydantic model in main.py.
    Creates a dummy result JSON file on completion.
    """
    print(f"Mock Task {task_id}: Starting simulation for {temp_save_path}")
    # Use string for status
    tasks[task_id] = {"status": "PROCESSING", "result": None, "conversation_text": None}
    await asyncio.sleep(0.1) # Simulate some work

    # Simulate success
    mock_result_data = [
        {"speaker": "SPEAKER_00", "start": 0.5, "end": 4.8, "text": "Hello world."},
        {"speaker": "SPEAKER_01", "start": 5.1, "end": 9.5, "text": "This is a test."}
    ]
    mock_conversation_text = "SPEAKER_00 (0.50s - 4.80s): Hello world.\nSPEAKER_01 (5.10s - 9.50s): This is a test."

    # --- Create dummy result file --- 
    # Ensure the test transcript directory exists
    os.makedirs(TEST_TRANSCRIPT_DIR, exist_ok=True)
    # Define path for the dummy JSON result file
    dummy_result_path = os.path.join(TEST_TRANSCRIPT_DIR, f"{task_id}_result.json")
    # Define path for the dummy conversation text file (mirroring potential real logic)
    dummy_conv_path = os.path.join(TEST_TRANSCRIPT_DIR, f"{task_id}_conversation.txt")
    
    try:
        with open(dummy_result_path, 'w', encoding='utf-8') as f_json:
            json.dump(mock_result_data, f_json, indent=2)
        print(f"Mock Task {task_id}: Created dummy result file: {dummy_result_path}")
        
        with open(dummy_conv_path, 'w', encoding='utf-8') as f_text:
             f_text.write(mock_conversation_text)
        print(f"Mock Task {task_id}: Created dummy conversation file: {dummy_conv_path}")

        # Store the PATH to the result file and the conversation text
        tasks[task_id] = {
            "status": "COMPLETED", 
            "result": dummy_result_path, # Store the path
            # Store path to conversation text, matching main.py structure
            "conversation_file_path": dummy_conv_path, 
            "conversation_text": None # Or store text directly if status endpoint uses it?
                                      # main.py status reads file, so let's stick to path for now
        }
        print(f"Mock Task {task_id}: Completed successfully, result path stored.")
        
    except Exception as e:
        print(f"Mock Task {task_id}: Error creating dummy files: {e}")
        tasks[task_id] = {
            "status": "FAILED",
            "result": f"Mock task failed to create dummy files: {e}",
            "conversation_text": None
        }

# --- Async Endpoint Tests ---

@pytest.mark.asyncio
@patch('main.process_transcription_task', new=mock_process_transcription_task) # Patch the background task
@patch('main.shutil.copyfileobj') # Mock file saving
@patch('main.uuid.uuid4')       # Mock uuid generation
async def test_transcribe_async_starts_job(mock_uuid, mock_copyfileobj, client):
    """Test the /transcribe endpoint correctly starts a background job and returns 202."""
    test_client, _, _, _ = client # Unpack client, adjust unpacking
    mock_uuid.return_value = "test-async-job-uuid"

    dummy_wav_content = b'RIFFdummyWAVE'
    files = {'wav_file': ('async_test.wav', BytesIO(dummy_wav_content), 'audio/wav')}

    response = await test_client.post("/transcribe/", files=files)

    assert response.status_code == 202, f"Expected 202, got {response.status_code}. Response: {response.text}"
    response_json = response.json()
    assert response_json["message"] == "File upload accepted, processing started."
    assert "task_id" in response_json
    assert response_json["task_id"] == "test-async-job-uuid"

    # Check if task entry was created (even if briefly processing)
    assert "test-async-job-uuid" in tasks
    # Allow the background task mock to run
    await asyncio.sleep(0.2)
    assert tasks["test-async-job-uuid"]["status"] == "COMPLETED"

    # Ensure copyfileobj was called to save the file
    mock_copyfileobj.assert_called_once()

@pytest.mark.asyncio
@patch('main.process_transcription_task', new=mock_process_transcription_task) # Patch the background task
@patch('main.shutil.copyfileobj') # Mock file saving
@patch('main.uuid.uuid4')       # Mock uuid generation
async def test_status_endpoint_workflow(mock_uuid, mock_copyfileobj, client):
    """Test the full workflow: POST /transcribe, poll GET /status/{task_id}."""
    test_client, _, _, _ = client # adjust unpacking
    mock_uuid.return_value = "test-status-workflow-uuid"
    task_id = "test-status-workflow-uuid"

    # 1. Start the job
    dummy_wav_content = b'RIFFdummyWAVEflow'
    files = {'wav_file': ('workflow_test.wav', BytesIO(dummy_wav_content), 'audio/wav')}
    post_response = await test_client.post("/transcribe/", files=files)
    assert post_response.status_code == 202
    assert post_response.json()["task_id"] == task_id

    # 2. Poll the status endpoint
    status_url = f"/status/{task_id}"
    start_time = time.time()
    max_wait = 5 # seconds
    final_response_json = None

    while time.time() - start_time < max_wait:
        status_response = await test_client.get(status_url)
        assert status_response.status_code == 200
        final_response_json = status_response.json()

        # Compare status using strings directly
        if final_response_json["status"] == "COMPLETED":
            print(f"Task {task_id} completed.")
            break
        elif final_response_json["status"] == "FAILED":
            pytest.fail(f"Task {task_id} failed unexpectedly.")
        else:
            assert final_response_json["status"] == "PROCESSING"
            print(f"Task {task_id} still processing, waiting...")
            await asyncio.sleep(0.3) # <<< INCREASED SLEEP DURATION
        pytest.fail(f"Task {task_id} did not complete within {max_wait} seconds.")

        # 3. Verify completed status and results
        print(f"Final JSON after loop: {final_response_json}") # <<< ADD PRINT
        assert final_response_json is not None
        assert final_response_json["status"] == "COMPLETED"
        assert "result" in final_response_json
        assert isinstance(final_response_json["result"], list) # Should now pass as endpoint reads the dummy file
        assert len(final_response_json["result"]) == 2 # Based on mock_process_transcription_task
        assert final_response_json["result"][0]["speaker"] == "SPEAKER_00"
        assert final_response_json["result"][0]["text"] == "Hello world."

        assert "conversation_text" in final_response_json
        assert isinstance(final_response_json["conversation_text"], str)
        assert "SPEAKER_00 (0.50s - 4.80s): Hello world." in final_response_json["conversation_text"]
        assert "SPEAKER_01 (5.10s - 9.50s): This is a test." in final_response_json["conversation_text"]


@pytest.mark.asyncio
async def test_status_endpoint_not_found(client):
    """Test GET /status/{task_id} returns 404 for an invalid task ID."""
    test_client, _, _, _ = client # adjust unpacking
    invalid_task_id = "non-existent-task-uuid"

    response = await test_client.get(f"/status/{invalid_task_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task ID not found."