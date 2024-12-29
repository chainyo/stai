# Silicon-Transcription AI

My own scripts to transcribe from a terminal some audio files.

## Pre-requisites

Check `ffmpeg` and `ffprobe` are installed and available.

```bash
ffmpeg -version
ffprobe -version
```

Install the env and get the whisper.cpp model.

```bash
uv sync
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
sh ./models/download-ggml-model.sh large-v3
cmake -B build -DWHISPER_COREML=1
cmake --build build -j --config Release
```

## Usage

* Transcribe a local audio file.

```bash
uv run stai transcribe --model large-v3 --file_path /path/to/audio.mp3
```

* Transcribe a Youtube video.

```bash
uv run stai transcribe --model large-v3 --url https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

* Transcribe a URL.

```bash
uv run stai transcribe --model large-v3 --url https://example.com/audio.wav
```
