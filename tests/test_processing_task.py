# tests/test_processing_task.py
import pytest
import os
import sys
import asyncio
from unittest.mock import patch, MagicMock, mock_open, call
import torch
import numpy as np
import json

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Import the function to test and the global tasks dict
from main import process_transcription_task, tasks, TRANSCRIPT_DIR

# --- Mock Data and Setup ---

MOCK_TASK_ID = "unit-test-task-123"
MOCK_TEMP_PATH = f"./uploads/{MOCK_TASK_ID}_upload.wav"
MOCK_RESULT_JSON_PATH = os.path.join(TRANSCRIPT_DIR, f"{MOCK_TASK_ID}.json")
MOCK_CONV_TXT_PATH = os.path.join(TRANSCRIPT_DIR, f"{MOCK_TASK_ID}_conversation_sentences.txt")

MOCK_APP_STATE = {
    'model': MagicMock(name="MockWhisperModel"),
    'embedding_pipeline': MagicMock(name="MockEmbeddingPipeline"),
    'device': 'cpu'
}

MOCK_WHISPER_RESULT = {
    "segments": [
        {
            "start": 0.1, "end": 2.5,
            "words": [
                {"start": 0.1, "end": 0.5, "text": "Hello", "confidence": 0.9},
                {"start": 0.6, "end": 1.2, "text": "world", "confidence": 0.95},
            ]
        },
        {
            "start": 3.0, "end": 5.0,
            "words": [
                {"start": 3.0, "end": 3.8, "text": "Test", "confidence": 0.8},
                {"start": 3.9, "end": 4.5, "text": "again", "confidence": 0.85},
            ]
        }
    ]
}

MOCK_LIBROSA_LOAD_RESULT = (np.random.rand(16000 * 5).astype(np.float32), 16000) # 5 seconds dummy audio
MOCK_EMBEDDING_RESULT = torch.rand(1, 192) # Example shape, adjust if needed
MOCK_FINAL_JSON_DATA = [
    {'start': 0.1, 'end': 0.5, 'text': 'Hello', 'speaker': 'SPEAKER_00', 'confidence': 0.9},
    {'start': 0.6, 'end': 1.2, 'text': 'world', 'speaker': 'SPEAKER_00', 'confidence': 0.95},
    {'start': 3.0, 'end': 3.8, 'text': 'Test', 'speaker': 'SPEAKER_01', 'confidence': 0.8},
    {'start': 3.9, 'end': 4.5, 'text': 'again', 'speaker': 'SPEAKER_01', 'confidence': 0.85}
]
MOCK_CONVERSATION_LINES = ["[00:00] SPEAKER_00: Hello world.", "[00:03] SPEAKER_01: Test again."]

# Mock audio data (just the array part for whisper.load_audio)
MOCK_AUDIO_ARRAY = MOCK_LIBROSA_LOAD_RESULT[0]

# --- Test Cases ---

@pytest.fixture(autouse=True)
def clear_tasks_dict():
    """Clears the global tasks dict before and after each test."""
    tasks.clear()
    yield
    tasks.clear()

# Fixture for embedding pipeline mock for cleaner tests
@pytest.fixture
def embedding_pipeline_mock():
    mock_pipeline = MagicMock(name="MockEmbeddingPipeline")
    # Configure the mock's return value for encode_batch
    mock_pipeline.encode_batch.return_value = MOCK_EMBEDDING_RESULT
    # Add any other methods needed by the process_transcription_task if necessary
    return mock_pipeline

@patch('main.convert_json_to_sentences_text', return_value=MOCK_CONVERSATION_LINES)
@patch('main.os.path.exists', return_value=True)
@patch('main.os.remove')
@patch('main.json.dump')
@patch('builtins.open', new_callable=mock_open)
@patch('main.smooth_speaker_labels', side_effect=lambda x, window_size: x)
@patch('main.librosa.load', return_value=MOCK_LIBROSA_LOAD_RESULT)
@patch('main.whisper.transcribe', return_value=MOCK_WHISPER_RESULT)
def test_process_task_success_restructured(
    mock_whisper_transcribe, mock_librosa_load, mock_smooth_labels, mock_file_open,
    mock_json_dump, mock_os_remove, mock_path_exists, mock_convert_text,
    embedding_pipeline_mock
):
    """Test successful execution with restructured patches."""
    mock_app_state_with_mock = MOCK_APP_STATE.copy()
    mock_app_state_with_mock['embedding_pipeline'] = embedding_pipeline_mock
    tasks[MOCK_TASK_ID] = {"status": "PENDING"}

    process_transcription_task(MOCK_TEMP_PATH, MOCK_TASK_ID, mock_app_state_with_mock)

    assert tasks[MOCK_TASK_ID]["status"] == "completed"
    mock_whisper_transcribe.assert_called_once()
    mock_librosa_load.assert_called_once_with(MOCK_TEMP_PATH, sr=16000, mono=True) # Verify diarization load
    assert embedding_pipeline_mock.encode_batch.call_count > 0
    mock_smooth_labels.assert_called_once()
    mock_json_dump.assert_called_once()
    mock_file_open.assert_has_calls([
        call(MOCK_RESULT_JSON_PATH, "w", encoding="utf-8"),
        call(MOCK_CONV_TXT_PATH, "w", encoding="utf-8")
    ], any_order=True)
    mock_path_exists.assert_called_with(MOCK_TEMP_PATH)
    mock_os_remove.assert_called_once_with(MOCK_TEMP_PATH)
    assert tasks[MOCK_TASK_ID]["result"] == MOCK_RESULT_JSON_PATH
    assert tasks[MOCK_TASK_ID]["conversation_file_path"] == MOCK_CONV_TXT_PATH

# Keep transcription failure test simple
@patch('main.whisper.transcribe', side_effect=Exception("Whisper Test Error"))
@patch('main.os.remove')
@patch('main.os.path.exists', return_value=True)
def test_process_task_transcription_failure(mock_path_exists, mock_os_remove, mock_whisper_transcribe):
    """Test failure during the Whisper transcription stage."""
    tasks[MOCK_TASK_ID] = {"status": "PENDING"}
    process_transcription_task(MOCK_TEMP_PATH, MOCK_TASK_ID, MOCK_APP_STATE)
    assert tasks[MOCK_TASK_ID]["status"] == "failed" # Lowercase
    assert "Whisper Test Error" in tasks[MOCK_TASK_ID]["result"]
    mock_path_exists.assert_called_once_with(MOCK_TEMP_PATH)
    mock_os_remove.assert_called_once_with(MOCK_TEMP_PATH)

# Test audio load failure for diarization
@patch('main.whisper.transcribe', return_value=MOCK_WHISPER_RESULT) # Assume transcription works
@patch('main.librosa.load', side_effect=Exception("Librosa Load Error")) # Mock diarization load failure
@patch('main.os.remove')
@patch('main.os.path.exists', return_value=True)
def test_process_task_diarization_audio_load_failure(mock_path_exists, mock_os_remove, mock_librosa_load, mock_whisper_transcribe):
    """Test failure during audio loading for diarization."""
    tasks[MOCK_TASK_ID] = {"status": "PENDING"}
    process_transcription_task(MOCK_TEMP_PATH, MOCK_TASK_ID, MOCK_APP_STATE)
    assert tasks[MOCK_TASK_ID]["status"] == "failed" # Lowercase
    assert "Librosa Load Error" in tasks[MOCK_TASK_ID]["result"]
    mock_path_exists.assert_called_once_with(MOCK_TEMP_PATH)
    mock_os_remove.assert_called_once_with(MOCK_TEMP_PATH)
    mock_whisper_transcribe.assert_called_once() # Ensure transcribe was called

# Test save failure (keep patches similar to success case)
@patch('main.convert_json_to_sentences_text', return_value=MOCK_CONVERSATION_LINES)
@patch('main.os.path.exists', return_value=True)
@patch('main.os.remove')
@patch('main.json.dump', side_effect=OSError("Cannot write JSON")) # Save failure
@patch('builtins.open', new_callable=mock_open)
@patch('main.smooth_speaker_labels', side_effect=lambda x, window_size: x)
@patch('main.librosa.load', return_value=MOCK_LIBROSA_LOAD_RESULT)
@patch('main.whisper.transcribe', return_value=MOCK_WHISPER_RESULT)
def test_process_task_save_failure_restructured(
     mock_whisper_transcribe, mock_librosa_load, mock_smooth_labels, mock_file_open,
     mock_json_dump, mock_os_remove, mock_path_exists, mock_convert_text,
     embedding_pipeline_mock
):
    """Test failure during saving the final JSON result."""
    embedding_pipeline_mock.encode_batch.return_value = MOCK_EMBEDDING_RESULT
    mock_app_state_with_mock = MOCK_APP_STATE.copy()
    mock_app_state_with_mock['embedding_pipeline'] = embedding_pipeline_mock
    tasks[MOCK_TASK_ID] = {"status": "PENDING"}

    process_transcription_task(MOCK_TEMP_PATH, MOCK_TASK_ID, mock_app_state_with_mock)

    assert tasks[MOCK_TASK_ID]["status"] == "failed" # Lowercase
    assert "Cannot write JSON" in tasks[MOCK_TASK_ID]["result"]
    assert call(MOCK_RESULT_JSON_PATH, "w", encoding="utf-8") in mock_file_open.call_args_list
    mock_path_exists.assert_called_with(MOCK_TEMP_PATH)
    mock_os_remove.assert_called_once_with(MOCK_TEMP_PATH)