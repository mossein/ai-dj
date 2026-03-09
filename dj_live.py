#!/usr/bin/env python3
"""
Live DJ Engine — Background audio player controlled via command file.

Claude writes commands → this engine reads them and plays audio in real-time.

Commands (written to dj_commands.json):
  {"cmd": "load", "track": "path.mp3", "position": "next"}
  {"cmd": "transition", "style": "smooth|hard|bass_swap|echo_out"}
  {"cmd": "transition_at", "bar": 41}  — transition at specific bar
  {"cmd": "stop"}
  {"cmd": "pause"}
  {"cmd": "resume"}
  {"cmd": "skip"}
  {"cmd": "volume", "level": 0.8}

Status (written to dj_status.json):
  {"state": "playing", "track": "track1.mp3", "position_s": 45.2,
   "next_track": "track2.mp3", "next_ready": true, ...}
"""

import os
import sys
import json
import hashlib
import subprocess
import tempfile
import shutil
import threading
import time
import traceback

import numpy as np
np.int = int
np.float = float
np.complex = complex

import sounddevice as sd
import soundfile as sf
import librosa
import madmom
import pyloudnorm
from scipy import signal as scipy_signal
from pedalboard import Pedalboard, Reverb, HighpassFilter, LowpassFilter

STEM_CACHE_DIR = "stem_cache"
CMD_FILE = "dj_commands.json"
STATUS_FILE = "dj_status.json"
QUEUE_FILE = "dj_queue.json"  # pending commands


# ============================================================
#  DSP UTILS (same as before, compact)
# ============================================================
def detect_beats(path):
    proc = madmom.features.beats.RNNBeatProcessor()(path)
    return madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(proc)

def detect_downbeats(path):
    proc = madmom.features.downbeats.RNNDownBeatProcessor()(path)
    r = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)(proc)
    return r[r[:, 1] == 1, 0]

def compute_bpm(beats):
    bpm = 60.0 / np.median(np.diff(beats))
    while bpm < 80: bpm *= 2
    while bpm > 200: bpm /= 2
    return round(bpm, 1)

def measure_lufs(audio, sr):
    meter = pyloudnorm.Meter(sr)
    if audio.ndim == 1: audio = np.stack([audio, audio])
    return meter.integrated_loudness(audio.T)

def get_file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()

def time_stretch(audio, sr, ratio):
    if abs(ratio - 1.0) < 0.002: return audio
    tmp = tempfile.mkdtemp(prefix="dj_rb_")
    try:
        sf.write(f"{tmp}/in.wav", audio.T, sr)
        subprocess.run(["rubberband", "-t", str(ratio), f"{tmp}/in.wav", f"{tmp}/out.wav"],
                      capture_output=True, check=True)
        s, _ = sf.read(f"{tmp}/out.wav", always_2d=True)
        return s.T
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def load_stems(path, target_sr=None):
    fh = get_file_hash(path)
    d = os.path.join(STEM_CACHE_DIR, fh)
    names = ['drums', 'bass', 'vocals', 'other']
    if os.path.isdir(d) and all(os.path.exists(f"{d}/{s}.wav") for s in names):
        stems = {}
        for n in names:
            a, fsr = sf.read(f"{d}/{n}.wav", always_2d=True); a = a.T
            if target_sr and fsr != target_sr:
                a = np.stack([librosa.resample(a[ch], orig_sr=fsr, target_sr=target_sr) for ch in range(a.shape[0])])
            stems[n] = a
        return stems
    return None

def run_demucs(path):
    """Run demucs and cache stems."""
    fh = get_file_hash(path)
    d = os.path.join(STEM_CACHE_DIR, fh)
    if os.path.isdir(d) and all(os.path.exists(f"{d}/{s}.wav") for s in ['drums','bass','vocals','other']):
        return  # already cached
    tmp = tempfile.mkdtemp(prefix="dj_stems_")
    name = os.path.splitext(os.path.basename(path))[0]
    subprocess.run([sys.executable, "-m", "demucs", "-n", "htdemucs", "--out", tmp, os.path.abspath(path)],
                  capture_output=True, check=True)
    os.makedirs(d, exist_ok=True)
    for s in ['drums','bass','vocals','other']:
        shutil.copy2(f"{tmp}/htdemucs/{name}/{s}.wav", f"{d}/{s}.wav")
    shutil.rmtree(tmp, ignore_errors=True)

def make_envelope(kp, n):
    env = np.zeros(n)
    for i in range(len(kp)-1):
        s0 = max(0, min(int(kp[i][0]*n), n))
        s1 = max(0, min(int(kp[i+1][0]*n), n))
        if s1 > s0: env[s0:s1] = np.linspace(kp[i][1], kp[i+1][1], s1-s0)
    ls = max(0, min(int(kp[-1][0]*n), n))
    env[ls:] = kp[-1][1]
    return np.clip(env, 0.0, 1.0)

def highpass(audio, sr, freq):
    if freq < 25: return audio
    sos = scipy_signal.butter(4, min(freq, sr*0.45), btype='high', fs=sr, output='sos')
    return scipy_signal.sosfilt(sos, audio, axis=-1)


# ============================================================
#  TRACK ANALYSIS
# ============================================================
class TrackInfo:
    """Holds all analysis data for a track."""
    def __init__(self, path, sr=48000):
        self.path = path
        self.sr = sr
        self.audio = None
        self.stems = None
        self.beats = None
        self.bars = None
        self.bpm = None
        self.lufs = None
        self.ready = False
        self.error = None

    def analyze(self):
        """Full analysis — call in background thread."""
        try:
            print(f"  [analyze] Loading {os.path.basename(self.path)}...")
            y, file_sr = sf.read(self.path, always_2d=True); y = y.T
            if file_sr != self.sr:
                y = np.stack([librosa.resample(y[ch], orig_sr=file_sr, target_sr=self.sr)
                              for ch in range(y.shape[0])])
            self.audio = y
            self.lufs = measure_lufs(y, self.sr)

            print(f"  [analyze] Beats...")
            self.beats = detect_beats(self.path)
            self.bars = detect_downbeats(self.path)
            self.bpm = compute_bpm(self.beats)

            print(f"  [analyze] Stems...")
            run_demucs(self.path)
            self.stems = load_stems(self.path, target_sr=self.sr)

            self.ready = True
            print(f"  [analyze] {os.path.basename(self.path)} ready: {self.bpm} BPM, {self.lufs:.1f} LUFS, {y.shape[1]/self.sr:.1f}s")
        except Exception as e:
            self.error = str(e)
            print(f"  [analyze] ERROR: {e}")
            traceback.print_exc()


# ============================================================
#  TRANSITION BUILDER
# ============================================================
def build_transition(track_a: TrackInfo, track_b: TrackInfo,
                     style="bass_swap", loop_bar_start=None, loop_bars=8,
                     n_repeats=3, b_cue_bar=None):
    """
    Build a transition buffer between two analyzed tracks.

    Args:
        style: "bass_swap", "smooth", "hard", "echo_out"
        loop_bar_start: which bar to start looping in Track A (0-indexed, auto if None)
        loop_bars: how many bars to loop
        n_repeats: how many times to repeat the loop
        b_cue_bar: which bar to cue Track B at (0-indexed, auto if None)

    Returns:
        dict with 'before_a', 'transition', 'after_b' buffers and metadata
    """
    sr = track_a.sr

    # --- Time stretch Track B stems to Track A's BPM ---
    stems_b = {}
    bars_b = track_b.bars.copy()
    rb_ratio = track_b.bpm / track_a.bpm

    if abs(rb_ratio - 1.0) > 0.002:
        print(f"  [transition] Stretching B: {track_b.bpm} → {track_a.bpm} BPM")
        for name in track_b.stems:
            stems_b[name] = time_stretch(track_b.stems[name], sr, rb_ratio)
        bars_b = bars_b * rb_ratio
    else:
        for name in track_b.stems:
            stems_b[name] = track_b.stems[name].copy()

    # LUFS match
    min_len = min(s.shape[1] for s in stems_b.values())
    y_b_full = sum(s[:, :min_len] for s in stems_b.values())
    b_lufs = measure_lufs(y_b_full, sr)
    if not np.isinf(b_lufs) and not np.isinf(track_a.lufs):
        b_gain = 10 ** ((track_a.lufs - b_lufs) / 20.0)
        for name in stems_b:
            stems_b[name] = stems_b[name] * b_gain
        y_b_full = y_b_full * b_gain
        print(f"  [transition] LUFS gain: {b_gain:.3f}")

    bars_a = track_a.bars
    stems_a = track_a.stems

    # --- Find loop point in Track A ---
    if loop_bar_start is None:
        # Auto: find energy dip, loop the groove before it
        y_mono = np.mean(track_a.audio, axis=0)
        hop = sr // 2
        rms = librosa.feature.rms(y=y_mono.astype(np.float32), frame_length=hop, hop_length=hop)[0]
        rms_norm = rms / (np.max(rms) + 1e-8)

        # Find biggest energy dip in 45-80% of track
        best_dip_idx = len(rms_norm) // 2
        best_dip_score = 0
        for i in range(16, len(rms_norm) - 16):
            pct = (i * 0.5) / (track_a.audio.shape[1] / sr)
            if not (0.35 < pct < 0.85): continue
            before = np.mean(rms_norm[max(0,i-16):i])
            if before > 0.4 and rms_norm[i] < before * 0.6:
                score = before - rms_norm[i]
                if score > best_dip_score:
                    best_dip_score = score
                    best_dip_idx = i

        dip_time = best_dip_idx * 0.5
        dip_bar = np.argmin(np.abs(bars_a - dip_time)) if len(bars_a) > 0 else len(bars_a) // 2
        loop_bar_start = max(0, dip_bar - loop_bars)
        print(f"  [transition] Auto loop: bars {loop_bar_start+1}-{loop_bar_start+loop_bars} (before dip at {dip_time:.1f}s)")

    loop_end_bar = min(loop_bar_start + loop_bars, len(bars_a) - 1)
    loop_start_s = float(bars_a[loop_bar_start])
    loop_end_s = float(bars_a[loop_end_bar])
    loop_start = int(loop_start_s * sr)
    loop_end = int(loop_end_s * sr)

    # Build loop
    xfade = min(int(0.03 * sr), (loop_end - loop_start) // 4)
    a_looped = {}
    for name in stems_a:
        end = min(loop_end, stems_a[name].shape[1])
        section = stems_a[name][:, loop_start:end]
        n_ch, loop_len = section.shape
        xf = min(xfade, loop_len // 4)
        eff = loop_len - xf
        total = eff * n_repeats + xf
        result = np.zeros((n_ch, total))
        for rep in range(n_repeats):
            off = rep * eff
            chunk = section.copy()
            if rep > 0: chunk[:, :xf] *= np.linspace(0, 1, xf)[np.newaxis, :]
            if rep < n_repeats - 1: chunk[:, -xf:] *= np.linspace(1, 0, xf)[np.newaxis, :]
            result[:, off:off+loop_len] += chunk
        a_looped[name] = result

    loop_n = min(s.shape[1] for s in a_looped.values())
    for name in a_looped:
        a_looped[name] = a_looped[name][:, :loop_n]

    # --- Track B cue ---
    if b_cue_bar is None:
        # Auto: find a high-energy, low-vocal section
        b_cue_bar = 0
        # Try to find a drop section (high energy, low mids)
        bar_dur = np.median(np.diff(bars_b)) if len(bars_b) > 1 else 2.0
        for i in range(0, len(bars_b) - 8, 8):
            start = int(bars_b[i] * sr)
            end = int(bars_b[min(i+8, len(bars_b)-1)] * sr)
            end = min(end, y_b_full.shape[1])
            if start >= end: continue
            chunk_rms = np.sqrt(np.mean(y_b_full[:, start:end] ** 2))
            if chunk_rms > 0.15:  # decent energy
                b_cue_bar = i
                break
        print(f"  [transition] Track B cue: bar {b_cue_bar+1} at {bars_b[b_cue_bar]:.1f}s")

    b_cue_bar = min(b_cue_bar, len(bars_b) - 1)
    b_cue_sample = int(bars_b[b_cue_bar] * sr)

    b_section = {}
    for name in stems_b:
        end = min(b_cue_sample + loop_n, stems_b[name].shape[1])
        seg = stems_b[name][:, b_cue_sample:end]
        if seg.shape[1] < loop_n:
            seg = np.concatenate([seg, np.zeros((seg.shape[0], loop_n - seg.shape[1]))], axis=1)
        b_section[name] = seg[:, :loop_n]

    n = loop_n

    # --- Build envelopes based on style ---
    swap_pct = 0.65

    if style == "bass_swap":
        env_a = {
            'drums': make_envelope([(0,1.0),(0.25,1.0),(0.4,0.8),(swap_pct,0.0)], n),
            'bass': make_envelope([(0,1.0),(swap_pct-0.01,1.0),(swap_pct,0.0)], n),
            'vocals': make_envelope([(0,0.8),(0.15,0.5),(0.3,0.0)], n),
            'other': make_envelope([(0,1.0),(0.2,0.8),(0.4,0.4),(0.6,0.15),(0.75,0.0)], n),
        }
        env_b = {
            'drums': make_envelope([(0,0.0),(0.05,0.12),(0.15,0.2),(0.3,0.35),(0.5,0.6),(swap_pct,0.85),(0.75,1.0)], n),
            'bass': make_envelope([(0,0.0),(swap_pct-0.01,0.0),(swap_pct,1.0)], n),
            'vocals': make_envelope([(0,0.0),(0.6,0.0),(swap_pct,0.0),(0.75,0.2),(0.9,0.8),(1.0,1.0)], n),
            'other': make_envelope([(0,0.0),(0.3,0.0),(0.4,0.1),(0.55,0.3),(swap_pct,0.6),(0.8,0.9),(0.9,1.0)], n),
        }
        # HP sweep on B drums
        hp_end = int(0.50 * n)
        n_chunks = 32
        chunk_size = hp_end // n_chunks
        freqs = np.linspace(6000, 80, n_chunks)
        bd = b_section['drums'][:, :hp_end].copy()
        for i in range(n_chunks):
            s0 = i * chunk_size
            s1 = s0 + chunk_size if i < n_chunks - 1 else hp_end
            if freqs[i] > 30: bd[:, s0:s1] = highpass(bd[:, s0:s1], sr, freqs[i])
        b_section['drums'][:, :hp_end] = bd

    elif style == "smooth":
        # Equal-power crossfade, no bass swap
        env_a = {s: make_envelope([(0,1.0),(0.3,0.9),(0.7,0.3),(1.0,0.0)], n) for s in ['drums','bass','vocals','other']}
        env_b = {s: make_envelope([(0,0.0),(0.3,0.1),(0.7,0.7),(1.0,1.0)], n) for s in ['drums','bass','vocals','other']}

    elif style == "hard":
        # Quick 4-bar swap
        cut = 0.3
        env_a = {s: make_envelope([(0,1.0),(cut-0.02,1.0),(cut,0.0)], n) for s in ['drums','bass','vocals','other']}
        env_b = {s: make_envelope([(0,0.0),(cut-0.02,0.0),(cut,1.0)], n) for s in ['drums','bass','vocals','other']}

    elif style == "echo_out":
        # Track A echoes out with heavy reverb, Track B fades in underneath
        env_a = {
            'drums': make_envelope([(0,1.0),(0.3,0.5),(0.5,0.0)], n),
            'bass': make_envelope([(0,1.0),(0.4,0.8),(0.6,0.0)], n),
            'vocals': make_envelope([(0,1.0),(0.2,0.7),(0.5,0.0)], n),
            'other': make_envelope([(0,1.0),(0.3,0.8),(0.7,0.2),(1.0,0.0)], n),
        }
        env_b = {
            'drums': make_envelope([(0,0.0),(0.2,0.2),(0.5,0.7),(0.7,1.0)], n),
            'bass': make_envelope([(0,0.0),(0.4,0.0),(0.5,0.5),(0.7,1.0)], n),
            'vocals': make_envelope([(0,0.0),(0.5,0.0),(0.7,0.5),(0.9,1.0)], n),
            'other': make_envelope([(0,0.0),(0.3,0.1),(0.5,0.4),(0.7,0.8),(0.9,1.0)], n),
        }

    elif style == "vocal_ride":
        # Keep Track A vocals riding over Track B's instrumental
        # Track A: vocals stay strong, everything else fades
        # Track B: full instrumental (no vocals), then B vocals take over at the end
        env_a = {
            'drums': make_envelope([(0,1.0),(0.2,0.7),(0.4,0.3),(0.55,0.0)], n),
            'bass': make_envelope([(0,1.0),(0.3,0.8),(swap_pct-0.01,0.5),(swap_pct,0.0)], n),
            'vocals': make_envelope([(0,1.0),(0.4,1.0),(0.6,0.9),(0.8,0.7),(0.9,0.3),(1.0,0.0)], n),
            'other': make_envelope([(0,1.0),(0.25,0.6),(0.5,0.2),(0.7,0.0)], n),
        }
        env_b = {
            'drums': make_envelope([(0,0.0),(0.1,0.2),(0.25,0.5),(0.4,0.8),(0.55,1.0)], n),
            'bass': make_envelope([(0,0.0),(swap_pct-0.01,0.0),(swap_pct,1.0)], n),
            'vocals': make_envelope([(0,0.0),(0.8,0.0),(0.9,0.5),(1.0,1.0)], n),  # B vocals only at very end
            'other': make_envelope([(0,0.0),(0.15,0.1),(0.3,0.3),(0.5,0.6),(0.7,0.9),(0.8,1.0)], n),
        }
        # HP sweep on B drums
        hp_end = int(0.40 * n)
        n_chunks = 32
        chunk_size = hp_end // n_chunks
        freqs = np.linspace(6000, 80, n_chunks)
        bd = b_section['drums'][:, :hp_end].copy()
        for i in range(n_chunks):
            s0 = i * chunk_size
            s1 = s0 + chunk_size if i < n_chunks - 1 else hp_end
            if freqs[i] > 30: bd[:, s0:s1] = highpass(bd[:, s0:s1], sr, freqs[i])
        b_section['drums'][:, :hp_end] = bd

    # --- Reverb wash on Track A 'other' ---
    rev_start = int(0.2 * n)
    rev_end = int(0.75 * n)
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=250),
        Reverb(room_size=0.85, damping=0.6, wet_level=0.4, dry_level=0.0),
        LowpassFilter(cutoff_frequency_hz=6000),
    ])
    a_other_rev = board(a_looped['other'][:, rev_start:rev_end].copy().astype(np.float32), sr)

    # --- Mix ---
    transition = np.zeros((2, n))
    for stem in ['drums', 'bass', 'vocals', 'other']:
        a_c = a_looped[stem] * env_a[stem][np.newaxis, :]
        b_c = b_section[stem] * env_b[stem][np.newaxis, :]
        if stem == 'other':
            rl = min(a_other_rev.shape[1], rev_end - rev_start)
            if rl > 0:
                rb = np.linspace(0, 0.6, rl)
                db = np.linspace(1, 0.4, rl)
                a_c[:, rev_start:rev_start+rl] = (
                    a_c[:, rev_start:rev_start+rl] * db[np.newaxis, :] +
                    a_other_rev[:, :rl] * rb[np.newaxis, :] * env_a['other'][rev_start:rev_start+rl][np.newaxis, :]
                )
        transition += a_c + b_c

    # Soft clip
    pk = np.max(np.abs(transition))
    if pk > 0.95:
        sc = 1.2 / pk
        transition = np.tanh(transition * sc) / np.tanh(sc)

    # --- Build final pieces ---
    a_before = track_a.audio[:, :loop_start]

    b_cont = b_cue_sample + loop_n
    b_after = y_b_full[:, b_cont:] if b_cont < y_b_full.shape[1] else np.zeros((2, sr))

    # Anti-click crossfades
    xfl = min(int(0.005 * sr), 512)
    if xfl > 0 and a_before.shape[1] > xfl and transition.shape[1] > xfl:
        fo = np.linspace(1, 0, xfl)[np.newaxis, :]
        fi = np.linspace(0, 1, xfl)[np.newaxis, :]
        blend = a_before[:, -xfl:] * fo + transition[:, :xfl] * fi
        a_before = np.concatenate([a_before[:, :-xfl], blend], axis=1)
        transition = transition[:, xfl:]
    if xfl > 0 and transition.shape[1] > xfl and b_after.shape[1] > xfl:
        fo = np.linspace(1, 0, xfl)[np.newaxis, :]
        fi = np.linspace(0, 1, xfl)[np.newaxis, :]
        blend = transition[:, -xfl:] * fo + b_after[:, :xfl] * fi
        transition = np.concatenate([transition[:, :-xfl], blend], axis=1)
        b_after = b_after[:, xfl:]

    return {
        'a_before': a_before,
        'transition': transition,
        'b_after': b_after,
        'loop_start_s': loop_start_s,
        'loop_end_s': loop_end_s,
        'b_cue_s': float(bars_b[b_cue_bar]),
        'swap_pct': swap_pct,
        'style': style,
        'trans_dur_s': n / sr,
    }


# ============================================================
#  LIVE DJ ENGINE
# ============================================================
class LiveDJ:
    def __init__(self):
        self.sr = 48000
        self.playing = False
        self.paused = False
        self.volume = 1.0
        self.position = 0
        self.buffer = None  # current playback buffer (2, N)
        self.stream = None

        self.current_track = None  # TrackInfo
        self.next_track = None     # TrackInfo
        self.next_analyzing = False
        self.next_ready = False

        self.transition_requested = False
        self.transition_style = "bass_swap"

        self.lock = threading.Lock()

        # Clean up old command/status files
        for f in [CMD_FILE, STATUS_FILE]:
            if os.path.exists(f): os.remove(f)

    def write_status(self):
        """Write current status for Claude to read."""
        status = {
            'state': 'paused' if self.paused else ('playing' if self.playing else 'stopped'),
            'current_track': self.current_track.path if self.current_track else None,
            'position_s': round(self.position / self.sr, 1) if self.buffer is not None else 0,
            'duration_s': round(self.buffer.shape[1] / self.sr, 1) if self.buffer is not None else 0,
            'next_track': self.next_track.path if self.next_track else None,
            'next_ready': self.next_ready,
            'next_analyzing': self.next_analyzing,
            'volume': self.volume,
            'transition_style': self.transition_style,
        }
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump(status, f, indent=2)
        except:
            pass

    def read_commands(self):
        """Check for new commands from Claude."""
        if not os.path.exists(CMD_FILE):
            return None
        try:
            with open(CMD_FILE, 'r') as f:
                cmd = json.load(f)
            os.remove(CMD_FILE)  # consume the command
            return cmd
        except:
            return None

    def load_track(self, path):
        """Load and analyze a track (blocking)."""
        info = TrackInfo(path, sr=self.sr)
        info.analyze()
        return info

    def load_next_background(self, path):
        """Analyze next track in background."""
        self.next_analyzing = True
        self.next_ready = False

        def _do():
            try:
                self.next_track = TrackInfo(path, sr=self.sr)
                self.next_track.analyze()
                self.next_ready = self.next_track.ready
            except Exception as e:
                print(f"  [error] Failed to load {path}: {e}")
            finally:
                self.next_analyzing = False

        t = threading.Thread(target=_do, daemon=True)
        t.start()

    def do_transition(self, style=None):
        """Execute transition from current to next track."""
        if not self.next_ready or self.next_track is None:
            print("  [!] Next track not ready")
            return

        if style:
            self.transition_style = style

        print(f"\n  [DJ] TRANSITION → {os.path.basename(self.next_track.path)} (style: {self.transition_style})")

        result = build_transition(
            self.current_track, self.next_track,
            style=self.transition_style,
        )

        # Build new buffer: current position to end of A → transition → B after
        with self.lock:
            # Keep what's playing up to now, append transition and Track B
            current_pos = self.position
            remaining_a = result['a_before']  # Track A up to loop point
            transition = result['transition']
            b_after = result['b_after']

            # New buffer from current position
            new_buf = np.concatenate([remaining_a, transition, b_after], axis=1)
            new_buf = np.clip(new_buf, -0.98, 0.98)

            self.buffer = new_buf
            self.position = 0  # restart from beginning of new buffer

        # Update current track to Track B
        self.current_track = self.next_track
        self.next_track = None
        self.next_ready = False
        self.transition_requested = False

        print(f"  [DJ] Transition built: {result['trans_dur_s']:.1f}s, style={result['style']}")

    def play_track(self, path):
        """Start playing a track."""
        print(f"\n  [DJ] Loading {os.path.basename(path)}...")
        self.current_track = self.load_track(path)
        if not self.current_track.ready:
            print("  [!] Failed to load track")
            return

        self.buffer = self.current_track.audio.copy()
        self.position = 0
        self.playing = True
        self.paused = False

    def audio_callback(self, outdata, frames, time_info, status):
        """sounddevice callback — streams audio."""
        if status:
            pass  # ignore xruns silently

        with self.lock:
            if self.buffer is None or self.paused:
                outdata[:] = 0
                return

            pos = self.position
            total = self.buffer.shape[1]
            end = min(pos + frames, total)
            n_frames = end - pos

            if n_frames <= 0:
                outdata[:] = 0
                self.playing = False
                return

            outdata[:n_frames] = (self.buffer[:, pos:end] * self.volume).T.astype(np.float32)
            if n_frames < frames:
                outdata[n_frames:] = 0

            self.position = end

    def run(self, initial_track=None):
        """Main loop — plays audio and watches for commands."""
        print("\n" + "=" * 60)
        print("  LIVE DJ ENGINE — Waiting for commands")
        print("  Write commands to dj_commands.json")
        print("=" * 60)

        if initial_track:
            self.play_track(initial_track)

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            blocksize=2048,
            callback=self.audio_callback,
            dtype='float32',
        )

        try:
            with self.stream:
                last_status = 0
                while True:
                    # Check for commands
                    cmd = self.read_commands()
                    if cmd:
                        self.handle_command(cmd)

                    # Write status every 0.5s
                    now = time.time()
                    if now - last_status > 0.5:
                        self.write_status()
                        last_status = now

                        # Print status line
                        if self.buffer is not None and self.playing:
                            pos = self.position
                            total = self.buffer.shape[1]
                            t = pos / self.sr
                            dur = total / self.sr
                            pct = pos / total * 100 if total > 0 else 0
                            filled = int(pct / 100 * 40)
                            bar = "█" * filled + "░" * (40 - filled)
                            track_name = os.path.basename(self.current_track.path) if self.current_track else "?"
                            next_name = os.path.basename(self.next_track.path) if self.next_track else "none"
                            next_status = "ready" if self.next_ready else ("analyzing..." if self.next_analyzing else "none")

                            sys.stdout.write(
                                f"\r  {bar} {t:5.1f}s/{dur:.0f}s | {track_name} | next: {next_status}    "
                            )
                            sys.stdout.flush()

                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n  [DJ] Stopped.")

    def handle_command(self, cmd):
        """Process a command dict."""
        action = cmd.get('cmd', '')
        print(f"\n  [cmd] {json.dumps(cmd)}")

        if action == 'load_next':
            path = cmd.get('track', '')
            if os.path.exists(path):
                self.load_next_background(path)
            else:
                print(f"  [!] File not found: {path}")

        elif action == 'play':
            path = cmd.get('track', '')
            if os.path.exists(path):
                self.play_track(path)

        elif action == 'transition':
            style = cmd.get('style', self.transition_style)
            self.do_transition(style=style)

        elif action == 'stop':
            self.playing = False
            self.buffer = None
            print("  [DJ] Stopped")

        elif action == 'pause':
            self.paused = True
            print("  [DJ] Paused")

        elif action == 'resume':
            self.paused = False
            print("  [DJ] Resumed")

        elif action == 'volume':
            self.volume = float(cmd.get('level', 1.0))
            print(f"  [DJ] Volume: {self.volume:.1f}")

        elif action == 'style':
            self.transition_style = cmd.get('style', 'bass_swap')
            print(f"  [DJ] Transition style: {self.transition_style}")

        else:
            print(f"  [!] Unknown command: {action}")


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    dj = LiveDJ()
    dj.run(initial_track=initial)
