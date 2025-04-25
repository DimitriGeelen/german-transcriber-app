#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipelines fail if any command fails, not just the last one.
set -o pipefail

# --- Configuration ---
PORT="8000" # The port your application runs on
HOST="127.0.0.1"
UPLOAD_DIR="./uploads"
TRANSCRIPT_DIR="./transcripts"
VENV_PYTHON="venv/bin/python"
VENV_UVICORN="venv/bin/uvicorn"

# --- Check for Virtual Environment ---
if [ ! -f "$VENV_UVICORN" ]; then
    echo "Error: Virtual environment activation or Uvicorn not found at $VENV_UVICORN."
    echo "Please ensure the virtual environment 'venv' exists and dependencies are installed."
    exit 1
fi

# --- 1. Find and Kill Existing Server --- 
echo "---> Checking for existing server on port $PORT..."
PIDS=$(lsof -ti tcp:$PORT || true) # Get PIDs listening on the port, ignore error if none found

if [ -n "$PIDS" ]; then
    echo "Found existing process(es) on port $PORT: $PIDS"
    echo "Killing process(es) forcefully (kill -9)..."
    # Use xargs to handle multiple PIDs if necessary
    echo "$PIDS" | xargs kill -9
    sleep 2 # Give the OS a moment to release the port
    echo "Process(es) killed."
else
    echo "No existing server found on port $PORT."
fi

# --- 2. Clear Caches and Output Directories ---
echo "---> Clearing caches and output directories..."

# Remove Python bytecode cache
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# Clear uploads and transcripts directories
echo "Clearing $UPLOAD_DIR and $TRANSCRIPT_DIR contents..."
rm -rf "$UPLOAD_DIR"
rm -rf "$TRANSCRIPT_DIR"
mkdir -p "$UPLOAD_DIR"
mkdir -p "$TRANSCRIPT_DIR"

echo "Caches and directories cleared."

# --- 3. Restart Server --- 
echo "---> Restarting Uvicorn server in the background..."

# Make sure ffmpeg is checked (using the check within main.py implicitly)
# Start the server in the background, redirecting output to a log file
"$VENV_UVICORN" main:app --host "$HOST" --port "$PORT" > uvicorn_startup.log 2>&1 &
UVICORN_PID=$! # Get the PID of the background process

echo "Server started with PID $UVICORN_PID. Waiting a few seconds for startup..."
sleep 5 # Wait for the server to initialize (adjust if needed)

# --- 4. Check if Server Process is Running ---
echo "---> Checking if server process (PID $UVICORN_PID) is still running..."

# Check if the process with UVICORN_PID exists
# kill -0 PID checks existence without sending a signal
if kill -0 $UVICORN_PID 2>/dev/null; then 
    echo "SUCCESS: Server process (PID $UVICORN_PID) is running."
    echo "You can try accessing it at http://$HOST:$PORT/"
    echo "Note: This script doesn't guarantee the server is fully initialized or responding to HTTP."
    # Optional: Perform a quick, non-fatal curl check just for info?
    # if curl --silent --head --max-time 2 "http://$HOST:$PORT/" > /dev/null; then
    #     echo "INFO: Server responded to a quick HTTP check."
    # else
    #     echo "WARN: Server process is running, but did not respond to a quick HTTP check (might still be starting)."
    # fi
    exit 0 # Exit successfully, leave the server running
else
    echo "ERROR: Server process (PID $UVICORN_PID) is NOT running."
    echo "The server likely failed to start. Please check logs or try starting manually:"
    echo "  '$VENV_UVICORN main:app --host $HOST --port $PORT'"
    exit 1 # Exit with error status
fi

# This line is now unreachable because of exits above, but good practice
# exit 0 