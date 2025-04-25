#!/bin/bash

# Installation script for the German WAV Transcriber Web App

echo "Starting installation..."
set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
ENV_EXAMPLE_FILE=".env_example"
ENV_FILE=".env"
PYTHON_CMD="python3" # Use python3 by default

# --- Check Prerequisites ---
echo "Step 1: Checking prerequisites..."

# Check for Python 3
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Error: $PYTHON_CMD command not found. Please install Python 3.8+ and ensure it's in your PATH."
    exit 1
fi
echo "✅ Python 3 found."

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null
then
    echo "Error: ffmpeg command not found. Please install ffmpeg and ensure it's in your PATH."
    echo "  (e.g., 'sudo apt update && sudo apt install ffmpeg' on Debian/Ubuntu)"
    exit 1
fi
echo "✅ ffmpeg found."

# --- Set up Virtual Environment ---
echo "Step 2: Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv $VENV_DIR
    echo "✅ Virtual environment created."
else
    echo "✅ Virtual environment '$VENV_DIR' already exists."
fi

# --- Install Dependencies ---
echo "Step 3: Installing Python dependencies from $REQUIREMENTS_FILE..."
# Activate the virtual environment for this command block (or run pip directly)
# Using direct path to pip for robustness in script execution
"$VENV_DIR/bin/pip" install --upgrade pip # Upgrade pip first
"$VENV_DIR/bin/pip" install -r $REQUIREMENTS_FILE
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check requirements.txt and network connection."
    echo "You might need to install Rust (https://rustup.rs/) if you see errors related to 'tokenizers'."
    exit 1
fi
echo "✅ Dependencies installed successfully."

# --- Set up Configuration File ---
echo "Step 4: Setting up configuration file..."
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE_FILE" ]; then
        echo "Copying $ENV_EXAMPLE_FILE to $ENV_FILE..."
        cp $ENV_EXAMPLE_FILE $ENV_FILE
        echo "✅ $ENV_FILE created."
        echo "ℹ️ IMPORTANT: Please review and edit the $ENV_FILE file to configure model sizes, directories, host, and port as needed."
    else
        echo "Warning: $ENV_EXAMPLE_FILE not found. Cannot create $ENV_FILE automatically."
        echo "Please create an $ENV_FILE manually before running the server."
    fi
else
    echo "✅ Configuration file $ENV_FILE already exists."
    echo "ℹ️ Please ensure it is configured correctly for your setup."
fi

# --- Final Instructions ---
echo ""
echo "----------------------------------------"
echo "✅ Installation steps completed!"
echo "----------------------------------------"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "2. (If needed) Review and edit the configuration in '$ENV_FILE'."
echo ""
echo "3. Start the application server:"
echo "   Using the provided script (recommended):"
echo "     ./restart_server.sh"
echo "   Or manually with uvicorn:"
echo "     uvicorn main:app --host \$(grep -E '^HOST=' .env | cut -d '=' -f2 || echo '127.0.0.1') --port \$(grep -E '^PORT=' .env | cut -d '=' -f2 || echo '8000')"
echo ""
echo "4. Access the application in your browser at the address provided by the server (usually http://127.0.0.1:8000 based on your .env file)."
echo "" 