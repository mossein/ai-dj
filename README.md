# AI DJ

Research project exploring different approaches to building an AI DJ system that creates professional-quality transitions between tracks — aiming to match or beat human DJs, Spotify, and Apple Music's crossfade algorithms.

## The Goal

Build a system that does what real DJs do on CDJs and 3-band mixers:
- **Loop** sections of the outgoing track to extend the runway
- **Layer stems** progressively (hi-hats → percussion → synths → bass)
- **Bass swap** on downbeats — instant low-end handoff
- **Avoid vocal collisions** — never two vocals at once
- **Maintain energy** — no dead zones during transitions
- **Respect harmony** — handle key clashes between tracks

## Approaches Explored

### v1 — LLM-Driven (`smart_dj_pro.py`)
Used **Google Gemini** as a creative director: upload both tracks, get a multimodal analysis, then have the LLM output a detailed JSON mix plan (stem envelopes, effects, cue points). A second LLM pass refines the plan as a "technical engineer."

**Result:** Gemini hallucinated timestamps and produced inconsistent plans. The transitions sounded like basic crossfades despite the complex pipeline.

### v2 — DJtransGAN (`smart_dj_pro_v2.py`)
Integrated [DJtransGAN](https://github.com/ChenPaulYip/DJtransGAN) (ICASSP 2022), a GAN trained on real DJ mixes to predict fader and EQ curves. We mapped its 4-band output to stem envelopes and fell back to Gemini when it failed.

**Result:** The model was trained on a specific EDM dataset and didn't generalize to arbitrary tracks. Curves were often nonsensical for our track pairs.

### v3 — Pure DSP + Algorithmic (`dj_v3.py`)
Threw out the AI/ML approaches and built a deterministic DSP pipeline:
- **madmom** RNN for beat/downbeat detection
- **demucs** (htdemucs) for 4-stem separation
- **rubberband** for time-stretching
- **Krumhansl-Kessler** key detection + Camelot wheel compatibility
- Energy analysis to find optimal mix-out points
- Stem-based envelopes with progressive layering
- Bass swap aligned to bar boundaries

**Result:** Much more consistent. The loop + stem layering technique sounds like an actual DJ mixing. Energy dips still occur mid-transition but are manageable.

### v4 — Claude as the DJ (`claude_dj_mix.py`)
Instead of generic heuristics, Claude deeply analyzed both specific tracks (structure, stems, energy profiles, vocal zones, spectral content) and crafted a **bespoke mix plan** tailored to the track pair. Every envelope, cue point, and effect choice was a deliberate creative decision based on the analysis.

**Result:** Best output so far. Smart Track B cue selection (skipping intros, finding instrumental drops), vocal collision prevention based on actual stem energy, harmonic clash mitigation (reverb wash to blur conflicting keys).

### v5 — Live DJ Web App (`dj_app.py`)
Flask-based web UI with real-time audio playback via `sounddevice`. Features:
- Play/pause/stop with live progress
- Queue next track (from file or YouTube download)
- Multiple transition styles: **Bass Swap**, **Vocal Ride**, **Smooth**, **Hard Cut**, **Echo Out**
- Background stem analysis while current track plays
- Claude controls the mix via API calls

## Tech Stack

| Component | Tool |
|-----------|------|
| Beat/downbeat detection | madmom (RNN) |
| Stem separation | demucs (htdemucs) |
| Time stretching | rubberband |
| Key detection | Krumhansl-Kessler + Camelot wheel |
| Loudness matching | pyloudnorm (LUFS) |
| Effects | pedalboard (Spotify), scipy |
| Web UI | Flask |
| Audio playback | sounddevice |
| Track download | yt-dlp |

## Key Findings

1. **LLMs are bad DJs** — They can talk about music but can't reliably produce precise, time-aligned mix plans from audio analysis. Hallucinated timestamps are a dealbreaker.

2. **Pre-trained models don't generalize** — DJtransGAN works for its training distribution but fails on arbitrary track pairs.

3. **DSP + musical rules work** — Beat-grid alignment, energy analysis, stem separation, and DJ heuristics (loop the groove, layer stems, bass swap on the 1) produce the most reliable results.

4. **Per-track analysis matters** — Generic envelopes work okay, but analyzing each track's structure (where are the vocals? where's the drop? what's the energy curve?) and making bespoke decisions produces significantly better transitions.

5. **Gain staging is critical** — LUFS matching + soft clipping beats peak normalization. A few clipped samples are inaudible; reducing the whole mix by 5dB is very audible.

## Usage

```bash
# Install dependencies
pip install librosa soundfile madmom pyloudnorm pedalboard demucs yt-dlp sounddevice flask
brew install rubberband

# Run the algorithmic mixer
python dj_v3.py track_a.mp3 track_b.mp3 -o mix.wav

# Run Claude's bespoke mix (for track1/track2 specifically)
python claude_dj_mix.py

# Run the web app
python dj_app.py --port 5555 --play track1.mp3

# Download a track from YouTube
python dj_downloader.py "https://youtube.com/watch?v=..." track_name
```

## What's Next

- Pitch shifting for harmonic correction (key clash mitigation)
- Smarter loop selection (rhythmic similarity matching)
- Multi-track continuous mix (full set, not just A→B)
- Waveform visualization in the web UI
- Claude API integration in the web app for conversational DJ control
