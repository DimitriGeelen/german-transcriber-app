# German WAV Transcriber Web App 🇩🇪🎙️➡️📝

A simple web application built with FastAPI and OpenAI's Whisper to transcribe German `.wav` audio files locally.

## Features

*   Upload `.wav` files via a simple web interface.
*   Transcribes audio using a locally run Whisper model (configurable size).
*   Specifically configured for German language transcription (configurable).
*   Saves transcriptions to text files on the server.
*   Application behavior is configurable via environment variables (`.env` file).
*   Includes basic automated tests (using `pytest`).

## Prerequisites

Before setting up the Python environment, ensure you have the following installed on your system:

1.  **Python 3.8+:** Check with `python3 --version`.
2.  **`pip` (Python package installer):** Usually comes with Python.
3.  **`venv` (Python virtual environment tool):** Usually comes with Python.
4.  **`ffmpeg`:** This is a crucial dependency for Whisper to process audio files. Installation methods vary by OS:
    *   **Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
    *   **Fedora:** `sudo dnf install ffmpeg`
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Windows:** Download from the official ffmpeg website or use a package manager like Chocolatey (`choco install ffmpeg`).
    *   Verify installation with `ffmpeg -version`.
5.  **`rustc` (Rust compiler):** May be required during the installation of the `tokenizers` dependency (part of `openai-whisper`). If `pip install` fails related to `tokenizers`, install Rust from <https://rustup.rs/>.

## Setup & Installation

1.  **Clone the repository (or place the files in a directory):**
    ```bash
    # git clone <repository_url> # If applicable
    cd <your_project_directory>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # Use python3 if python is not linked
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows Command Prompt
    # .\venv\Scripts\Activate.ps1 # Windows PowerShell
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This step downloads FastAPI, Uvicorn, Whisper, Pytest, etc. It may take some time, especially for Whisper and its dependencies like PyTorch).* 

4.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env_example .env
        ```
    *   Edit the `.env` file. See the Configuration section below for details.

## Configuration (`.env` file)

Modify the `.env` file to control the application's behavior:

*   `WHISPER_MODEL_SIZE`: Specifies the Whisper model to use (e.g., `tiny`, `base`, `small`, `medium`, `large`). Smaller models are faster and use less memory but are less accurate. Larger models are more accurate but require more resources (RAM, potentially GPU) and take longer. Defaults to `base`.
*   `SUPPORTED_LANGUAGE`: The language Whisper should expect. Defaults to `german`.
*   `UPLOAD_DIR`: The directory where temporary uploaded `.wav` files are stored before transcription. Defaults to `./uploads`.
*   `TRANSCRIPT_DIR`: The directory where the final `.txt` transcript files are saved. Defaults to `./transcripts`.
*   `HOST`: The IP address the development server binds to (e.g., `127.0.0.1` for local access only, `0.0.0.0` to be accessible on your network). Defaults to `127.0.0.1`.
*   `PORT`: The port the development server listens on. Defaults to `8000`.

*Security Note:* Ensure `UPLOAD_DIR` and `TRANSCRIPT_DIR` are secure locations and the application is not exposed to untrusted networks if using `HOST=0.0.0.0`.

## Running the Application

1.  **Ensure your virtual environment is activated.** (`source venv/bin/activate`)
2.  **Start the FastAPI development server:**
    ```bash
    uvicorn main:app --reload --host $(grep -E '^HOST=' .env | cut -d '=' -f2 || echo '127.0.0.1') --port $(grep -E '^PORT=' .env | cut -d '=' -f2 || echo '8000')
    # Or simply use the defaults if HOST/PORT not set in .env:
    # uvicorn main:app --reload
    ```
    *   The `--reload` flag automatically restarts the server on code changes.
    *   The server address (e.g., `http://127.0.0.1:8000`) will be shown in the terminal.
    *   **First Run:** The first time you run the app (or transcribe), Whisper will download the specified model (`WHISPER_MODEL_SIZE`). This download can take time depending on the model size and your internet connection.

3.  **Access the Web Interface:**
    *   Open your web browser and navigate to the address shown by Uvicorn (e.g., `http://127.0.0.1:8000`).

4.  **Upload and Transcribe:**
    *   Use the form to select a German `.wav` file.
    *   Click "Transcribe".
    *   Wait for processing. A loading message will appear. **Transcription can take significant time**, especially for large files or models on less powerful hardware.
    *   A success message will appear with the filename of the saved transcript.

5.  **Find Transcripts:**
    *   Transcripts are saved as `.txt` files in the `TRANSCRIPT_DIR` (default: `./transcripts`).

## Running with Docker (Alternative)

This project includes a `Dockerfile` to build and run the application in a container. This is useful for creating a consistent environment.

**Prerequisites:**

*   Docker installed and running on your system.

**Building the Docker Image:**

1.  Navigate to the project's root directory (where the `Dockerfile` is located).
2.  Run the build command:
    ```bash
    docker build -t german-transcriber-app .
    ```
    *   This will download the base image, install dependencies (including `ffmpeg` and Python packages), and copy the application code into the image.
    *   The first build might take a while due to downloads and installations.

**Running the Docker Container:**

1.  **Create directories for uploads and transcripts on your host machine:** These will be mounted into the container so that the files persist even if the container is removed.
    ```bash
    mkdir -p ./host_uploads
    mkdir -p ./host_transcripts
    ```

2.  **Run the container, mounting the volumes and passing the `.env` file:**
    ```bash
    docker run -d --rm \
      -p 8000:8000 \
      --env-file .env \
      -v $(pwd)/host_uploads:/app/uploads \
      -v $(pwd)/host_transcripts:/app/transcripts \
      --name german-transcriber \
      german-transcriber-app
    ```
    *   `-d`: Run in detached mode (in the background).
    *   `--rm`: Automatically remove the container when it stops.
    *   `-p 8000:8000`: Map port 8000 on your host to port 8000 in the container.
    *   `--env-file .env`: Pass the environment variables from your local `.env` file to the container. **Important:** Ensure your `.env` file exists and is configured correctly *before* running this command.
    *   `-v $(pwd)/host_uploads:/app/uploads`: Mount the `host_uploads` directory on your host to `/app/uploads` inside the container (adjust path if needed).
    *   `-v $(pwd)/host_transcripts:/app/transcripts`: Mount the `host_transcripts` directory to `/app/transcripts`.
    *   `--name german-transcriber`: Assign a name to the container for easier management.
    *   `german-transcriber-app`: The name of the image built previously.

3.  **Access the Web Interface:**
    *   Open your browser and navigate to `http://localhost:8000` (or the host IP if Docker is running remotely).

4.  **Check Transcripts:**
    *   Uploaded files will appear temporarily inside the container's `/app/uploads` (and thus in your `host_uploads` directory).
    *   Completed transcripts will be saved to `/app/transcripts` inside the container (and thus in your `host_transcripts` directory).

5.  **View Logs:**
    ```bash
    docker logs german-transcriber
    ```

6.  **Stop the Container:**
    ```bash
    docker stop german-transcriber
    ```

## Running Tests

This project includes automated tests using `pytest`.

1.  **Ensure the virtual environment is activated and dev dependencies are installed.** (`pip install -r requirements.txt`)
2.  **Run tests:**
    ```bash
    PYTHONPATH=. pytest
    ```
    *   `PYTHONPATH=.` is needed so pytest can find the `main.py` module.
    *   Tests currently *mock* the Whisper transcription process to focus on API behavior (file handling, responses, cleanup).

## Notes & Potential Improvements

*   **Performance:** Transcription is CPU/GPU intensive. Consider using smaller models (`tiny`, `base`) for faster results on less powerful machines. GPU acceleration significantly speeds up Whisper if available and PyTorch is installed with CUDA support.
*   **Error Handling:** Basic error handling is in place. Check the terminal where `uvicorn` is running for detailed logs and error messages.
*   **File Type Validation:** The application currently only logs a warning for non-`.wav` files. Stricter validation could be added in `main.py`.
*   **Asynchronous Transcription:** For production use or handling many requests, the `model.transcribe` call (which is blocking) should ideally be run in a separate thread or process using tools like `FastAPI Background Tasks` or a dedicated task queue (e.g., Celery) to avoid blocking the server.
*   **Scalability:** This setup is intended for local/single-user operation. Scaling would require architectural changes. 