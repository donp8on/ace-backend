import os
import tempfile
import subprocess
import shutil
import urllib.request
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

SUPPORTED_SOURCES = ["soundcloud.com", "soundcloud.app", "on.soundcloud.com", ".mp3", ".wav", ".m4a", ".ogg", ".flac", "beatstars.com", "audiomack.com"]


class AnalyzeRequest(BaseModel):
    url: str


class AnalyzeResponse(BaseModel):
    bpm: float
    bpm_rounded: int
    key: str
    mode: str
    key_full: str
    confidence: float
    source: str


def resolve_url(url: str) -> str:
    """Follow redirects to get the final URL — handles on.soundcloud.com and other shorteners."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15"},
            method="HEAD"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.url
    except Exception:
        try:
            # fallback — GET request
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.url
        except Exception:
            return url  # return original if resolution fails


def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found
    for c in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/bin/ffmpeg"]:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def detect_source(url: str) -> str:
    if "soundcloud.com" in url or "soundcloud.app" in url or "on.soundcloud.com" in url:
        return "soundcloud"
    if "beatstars.com" in url:
        return "beatstars"
    if "audiomack.com" in url:
        return "audiomack"
    for ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
        if ext in url.lower():
            return "direct"
    return "unknown"


def analyze_audio(wav_file: str):
    y, sr = librosa.load(wav_file, sr=22050, mono=True, duration=90)

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

    return {
        "bpm": round(bpm, 1),
        "bpm_rounded": round(bpm),
        "key": KEY_NAMES[best_key],
        "mode": MODE_NAMES[best_mode],
        "key_full": f"{KEY_NAMES[best_key]} {MODE_NAMES[best_mode]}",
        "confidence": round(float((best_corr + 1) / 2), 2)
    }


@app.get("/")
def root():
    return {"status": "ACE Beat Analyzer running"}


@app.get("/resolve")
def resolve(url: str):
    """Debug endpoint — shows what URL a shortened link resolves to."""
    resolved = resolve_url(url)
    return {"original": url, "resolved": resolved}


@app.get("/health")
def health():
    ffmpeg = find_ffmpeg()
    return {"status": "ok", "ffmpeg": ffmpeg or "not found", "supported": SUPPORTED_SOURCES}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    url = req.url.strip()

    # Auto-resolve shortened/redirect URLs (on.soundcloud.com, bit.ly, etc.)
    url = resolve_url(url)

    source = detect_source(url)

    if source == "unknown":
        raise HTTPException(
            status_code=400,
            detail="Unsupported URL. Paste a SoundCloud link, BeatStars link, Audiomack link, or a direct audio file URL (.mp3, .wav, .m4a)"
        )

    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise HTTPException(status_code=500, detail="ffmpeg not found on server")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "beat.%(ext)s")

        if source == "direct":
            # Direct download for raw audio file URLs
            ext = next((e for e in [".mp3", ".wav", ".m4a", ".ogg", ".flac"] if e in url.lower()), ".mp3")
            dl_path = os.path.join(tmpdir, f"beat{ext}")
            try:
                req_obj = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req_obj, timeout=60) as resp:
                    with open(dl_path, "wb") as f:
                        f.write(resp.read())
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Could not download audio file: {str(e)}")

            # Convert to wav using ffmpeg
            wav_path = os.path.join(tmpdir, "beat.wav")
            conv = subprocess.run(
                [ffmpeg_path, "-i", dl_path, "-ar", "22050", "-ac", "1", wav_path],
                capture_output=True, text=True, timeout=60
            )
            if conv.returncode != 0 or not os.path.isfile(wav_path):
                raise HTTPException(status_code=422, detail="Audio conversion failed")
            wav_file = wav_path

        else:
            # Use yt-dlp for SoundCloud, BeatStars, Audiomack — they don't block servers
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--max-filesize", "100m",
                "--ffmpeg-location", os.path.dirname(ffmpeg_path),
                "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1",
                "-o", audio_path,
                url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

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
            result_data = analyze_audio(wav_file)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Analysis failed: {str(e)}")

        return AnalyzeResponse(**result_data, source=source)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
