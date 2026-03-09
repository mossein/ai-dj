# DJ Killer: AI-First Smart DJ MVP (Progress Report - March 2026)

## 🎯 Vision
Build a "Spotify/Apple Music DJ Killer" that performs professional-grade transitions, active remixing, and vibe-aware track selection using Gemini 3.1 Pro/Flash.

---

## 🛠 Tech Stack (2026 Standards)
- **AI Brain:** `google-genai` SDK (Unified Client)
- **Models:** 
    - `gemini-3.1-pro-preview` (Master Producer / Strategic Reasoning)
    - `gemini-3.1-flash-lite-preview` (Ultra-low latency audio/structural analysis)
- **Audio Engine:** `librosa` (BPM/Beat detection), `pydub` (Surgical audio manipulation/filtering), `scipy` (Signal processing).
- **Orchestration:** `langgraph` (Agentic state-machine for DJing workflows).
- **Sourcing:** `yt-dlp` (High-quality YouTube-to-MP3 downloader).

---

## 🚀 Progress & Iterations

### 1. The "Math DJ" (Failure)
- **Logic:** Simple BPM matching + linear crossfade.
- **Outcome:** Sounded amateur; beats drifted, and tracks "muddy" when mixed.

### 2. The "Elite Pro" (Partial Success)
- **Logic:** "Bass Swap" (Frequency-split crossover). Sharp cut of Track A's bass as Track B's bass enters.
- **Outcome:** Better, but "clashing highs" occurred when both tracks were at peak energy.

### 3. The "Master Producer" (Current State)
- **Logic:** Gemini 3.1 Pro acts as lead producer, deciding "Master BPM" and "Bar-aligned" entry points.
- **Outcome:** Beats are perfectly locked via **Phase Matching** (aligning the 'One'), but the AI still doesn't "hear" the emotional context, leading to melodic clashing.

---

## 🧠 Critical Findings (Why it's not "Mind-Blowing" yet)
- **Energy Clashing:** The AI calculates BPM math but doesn't perceive "energy peaks." If Song A's vocal peak hits at the same time as Song B's synth lead, the mix sounds "very bad."
- **Phase vs. BPM:** BPM is just speed. **Phase** is the pulse. We successfully implemented Phase-alignment using Librosa's nearest-beat logic.
- **The "Listening" Gap:** To reach Spotify-Killer status, the AI must **Natively Listen** to the audio files (via Multimodal API) to "feel" the peaks and valleys before rendering.

---

## 📍 Next Session Goal: The "Listening DJ"
Implement the `listener_node` in the LangGraph:
1.  **Upload Audio:** Send raw MP3s to Gemini.
2.  **Structural Perception:** Ask Gemini: *"Where are the competing melodies? Where is the energy drop in Song A?"*
3.  **Active Stem Mixing:** Use Gemini's insights to perform surgical volume ducking on specific frequency bands where clashing is detected.

---

## 📄 Current Script: `smart_dj_pro.py`
*(The full code is preserved in the file system as `smart_dj_pro.py` and `dj_downloader.py`.)*
