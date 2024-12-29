"""Entrypoint for the Silicon-Transcription AI CLI."""

import os
import subprocess
from enum import Enum
from pathlib import Path

import requests
import typer
from yt_dlp import YoutubeDL


WHISPER_CPP_PATH = "./whisper.cpp"
PREFIX_WHISPER_CPP_CLI = f"{WHISPER_CPP_PATH}/build/bin/whisper-cli"

app = typer.Typer(name="Silicon-Transcription AI", no_args_is_help=True)


class TranscriptionMode(str, Enum):
    """The mode of transcription."""

    FILE = "file"
    URL = "url"
    YOUTUBE = "youtube"
    NOT_DETERMINED = "not determined"


@app.command()
def transcribe(
    model: str = typer.Option(..., help="The model to use for transcription."),
    file_path: str | None = typer.Option(None, help="The path to the audio file to transcribe."),
    url: str | None = typer.Option(None, help="The URL of the audio file to transcribe (Youtube or other URL)."),
) -> None:
    """Transcribe an audio file."""
    _mode = TranscriptionMode.NOT_DETERMINED
    if file_path:
        _mode = TranscriptionMode.FILE
    elif url:
        if url.startswith("https://www.youtube.com/watch?v=") or url.startswith("https://youtu.be/"):
            _mode = TranscriptionMode.YOUTUBE
        else:
            _mode = TranscriptionMode.URL
    else:
        typer.echo("Either file_path or youtube_url must be provided.", err=True)
        return

    models = _get_whisper_models()
    if model not in models:
        typer.echo(f"Model not found. Available models: {models}", err=True)
        return

    if _mode == TranscriptionMode.YOUTUBE:
        audio_path = _download_file_from_youtube(url, "yt_audio")
    elif _mode == TranscriptionMode.URL:
        audio_path = _download_file_from_url(url, "url_audio")
    else:
        audio_path = file_path

    result = subprocess.run(
        f"{PREFIX_WHISPER_CPP_CLI} -m {WHISPER_CPP_PATH}/models/ggml-{model}.bin -f {audio_path}",
        capture_output=True,
        text=True,
        shell=True,
    )

    transcription_lines: list[str] = []
    for line in result.stdout.split("\n"):
        if line.startswith("[") and "-->" in line:
            transcription_lines.append(line.split("]")[-1].strip())

    transcription = "\n".join(transcription_lines)
    with open(f"{audio_path}.txt", "w") as f:
        f.write(transcription)

    typer.echo(f"Transcription saved to {audio_path}.txt")


@app.command()
def download_model(model: str) -> None:
    """Download a whisper model."""
    models = _get_whisper_models()
    if model not in models:
        typer.echo(f"Model not found. Available models: {models}", err=True)
        return

    download_script = f"{WHISPER_CPP_PATH}/models/download-ggml-model.sh"
    try:
        result = subprocess.run([download_script, model], check=True)
        if result.returncode == 0:
            # If download successful, run the coreml preparation command
            typer.echo(f"Successfully downloaded {model}, preparing CoreML model...")
            coreml_result = subprocess.run([f"{WHISPER_CPP_PATH}/models/generate-coreml-model.sh {model}"], check=True)
            if coreml_result.returncode == 0:
                typer.echo("CoreML model preparation completed successfully")
            else:
                typer.echo("CoreML model preparation failed", err=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Command failed with return code {e.returncode}", err=True)


@app.command()
def list_models() -> None:
    """List the available whisper models."""
    models = _get_whisper_models()
    for model in models:
        if Path(f"{WHISPER_CPP_PATH}/models/ggml-{model}.bin").exists():
            typer.echo(f"✅ {model}")
        else:
            typer.echo(f"❌ {model}")


def _get_whisper_models() -> list[str]:
    """Get the list of available whisper models."""
    with open(f"{WHISPER_CPP_PATH}/models/download-ggml-model.sh", "r") as file:
        content = file.read()

    # Find the models section
    start = content.find('models="') + 8  # Skip 'models="'
    end = content.find('"', start)        # Find the closing quote
    
    # Extract and split the models string
    models_text = content[start:end]
    models_list = models_text.strip().split('\n')
    
    return models_list


def _download_file_from_youtube(url: str, filename: str) -> str:
    """
    Download a file from YouTube using youtube-dl.

    Args:
        url (str): URL of the YouTube video.
        filename (str): Filename to save the file as.

    Returns:
        str: Path to the downloaded file.
    """
    with YoutubeDL(
        {
            "format": "bestaudio",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "postprocessor_args": [
                "-ar", "16000",
                "-ac", "1",
            ],
            "outtmpl": f"{filename}",
            "quiet": True,
        }
    ) as ydl:
        ydl.download([url])

    return filename + ".wav"


def _download_file_from_url(
    url: str, filename: str, url_headers: dict[str, str] | None = None
) -> str:
    """
    Download a file from a URL using requests.

    Args:
        url (str): URL of the audio file.
        filename (str): Filename to save the file as.
        url_headers (dict[str, str] | None): Headers to send with the request. Defaults to None.

    Returns:
        str: Path to the downloaded file.

    Raises:
        Exception: If the file failed to download.
    """
    url_headers = url_headers or {}

    response = requests.get(url, headers=url_headers, stream=True)

    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    else:
        raise Exception(f"Failed to download file. Status: {response.status_code}")

    return filename