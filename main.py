import os
import tempfile
import shutil
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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


def analyze_audio(file_path: str) -> dict:
    # Load audio — librosa handles MP3, WAV, M4A, FLAC etc via ffmpeg
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=120)

    # BPM — use librosa's beat tracker with dynamic compression for accuracy
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(tempo)

    # Normalize BPM to 60-180 range
    while bpm < 60:  bpm *= 2
    while bpm > 180: bpm /= 2

    # Key detection — use harmonic component for cleaner chroma
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=36)
    # Use chroma energy normalized statistics for robustness
    chroma_mean = np.mean(chroma, axis=1)
    chroma_mean = chroma_mean / (chroma_mean.max() + 1e-6)

    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

    # Normalize profiles
    major_profile = major_profile / major_profile.sum()
    minor_profile = minor_profile / minor_profile.sum()

    best_corr = -2.0
    best_key = 0
    best_mode = 0

    for i in range(12):
        for mode_idx, profile in enumerate([minor_profile, major_profile]):
            rotated = np.roll(profile, i)
            corr = np.corrcoef(chroma_mean, rotated)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_key = i
                best_mode = mode_idx

    confidence = round(float((best_corr + 1) / 2), 2)

    return {
        "bpm": round(bpm, 1),
        "bpm_rounded": int(round(bpm)),
        "key": KEY_NAMES[best_key],
        "mode": MODE_NAMES[best_mode],
        "key_full": f"{KEY_NAMES[best_key]} {MODE_NAMES[best_mode]}",
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"status": "ACE Beat Analyzer running"}


@app.get("/health")
def health():
    ffmpeg = find_ffmpeg()
    return {"status": "ok", "ffmpeg": ffmpeg or "not found"}


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...)):
    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".m4a", ".aiff", ".aif", ".flac", ".ogg"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use MP3, WAV, M4A, AIFF, or FLAC."
        )

    # Check file size (50MB max)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB.")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, f"beat{ext}")
        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            result = analyze_audio(file_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Analysis failed: {str(e)}")

    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
