# ACE Beat Analyzer — Backend

A FastAPI server that accepts a YouTube URL, downloads the audio, and returns the BPM and musical key.

## Deploy to Railway (free)

1. Go to railway.app and sign up with GitHub
2. Click **New Project → Deploy from GitHub repo**
3. Push this folder to a GitHub repo first, then connect it
   OR use the Railway CLI:
   ```bash
   npm install -g @railway/cli
   railway login
   railway init
   railway up
   ```
4. Railway will build and deploy automatically
5. Once deployed, copy your URL — it looks like:
   `https://ace-backend-production.up.railway.app`

## Add the URL to iOS

Open `BeatAnalysisView.swift` in Xcode and replace:
```swift
let BACKEND_URL = "https://your-railway-url.up.railway.app"
```
with your actual Railway URL.

## Test it

```bash
curl -X POST https://your-url.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=EXAMPLE"}'
```

Expected response:
```json
{
  "bpm": 94.5,
  "bpm_rounded": 95,
  "key": "F#",
  "mode": "minor",
  "key_full": "F# minor",
  "confidence": 0.82
}
```

## Endpoints

- `GET /` — health check
- `GET /health` — health check
- `POST /analyze` — analyze a YouTube URL
  - Body: `{"url": "https://youtube.com/..."}`
  - Returns: BPM, key, mode, confidence
