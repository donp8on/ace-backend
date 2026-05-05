import os
import tempfile
import subprocess
import shutil
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import librosa
import uvicorn

app = FastAPI(title="ACE Beat Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MODE_NAMES = ["minor", "major"]


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    bpm: float
    bpm_rounded: int
    key: str
    mode: str
    key_full: str
    confidence: float


def find_ffmpeg():
    for candidate in ["ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/nix/var/nix/profiles/default/bin/ffmpeg"]:
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.isfile(candidate):
            return candidate
    return None


@app.get("/")
def root():
    return {"status": "ACE Beat Analyzer running"}


@app.get("/health")
def health():
    ffmpeg = find_ffmpeg()
    return {"status": "ok", "ffmpeg": ffmpeg or "not found"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    if not any(x in url for x in ["youtube.com", "youtu.be"]):
        raise HTTPException(status_code=400, detail="Only YouTube URLs are supported")

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise HTTPException(status_code=500, detail="ffmpeg not installed on server")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "beat.%(ext)s")

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--max-filesize", "50m",
            "--ffmpeg-location", ffmpeg_path,
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "--add-header", "Accept-Language:en-US,en;q=0.9",
            "--extractor-args", "youtube:player_client=web",
            "--no-check-certificates",
            "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1",
            "-o", audio_path,
            url
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "Download failed")[-600:]
            raise HTTPException(status_code=422, detail=f"Could not download audio: {detail}")

        wav_file = next(
            (os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".wav")),
            None
        )
        if not wav_file:
            raise HTTPException(status_code=422, detail="Audio extraction failed")

        try:
            y, sr = librosa.load(wav_file, sr=22050, mono=True, duration=90)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Audio load failed: {str(e)}")

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        best_corr = -2.0
        best_key = 0
        best_mode = 0

        for i in range(12):
            for mode_idx, profile in enumerate([minor_profile, major_profile]):
                corr = np.corrcoef(chroma_mean, np.roll(profile, i))[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_key = i
                    best_mode = mode_idx

        key_name = KEY_NAMES[best_key]
        mode_name = MODE_NAMES[best_mode]

        return AnalyzeResponse(
            bpm=round(bpm, 1),
            bpm_rounded=round(bpm),
            key=key_name,
            mode=mode_name,
            key_full=f"{key_name} {mode_name}",
            confidence=round(float((best_corr + 1) / 2), 2)
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
