#!/usr/bin/env python3
"""
Real-Time AI DJ — Plays the mix LIVE through your speakers.

Pre-loads cached stems and beat analysis, then streams audio in real-time
with stem-based transitions, looping, bass swaps — all live.

Usage: python realtime_dj.py [track_a.mp3] [track_b.mp3]
"""

import os
import sys
import hashlib
import subprocess
import tempfile
import shutil
import threading
import time

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


# ============================================================
#  ANALYSIS (cached/fast)
# ============================================================
def detect_beats(audio_path):
    proc = madmom.features.beats.RNNBeatProcessor()(audio_path)
    return madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(proc)

def detect_downbeats(audio_path):
    proc = madmom.features.downbeats.RNNDownBeatProcessor()(audio_path)
    result = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], fps=100)(proc)
    return result[result[:, 1] == 1, 0]

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

def time_stretch(audio, sr, time_ratio):
    if abs(time_ratio - 1.0) < 0.002: return audio
    tmp_dir = tempfile.mkdtemp(prefix="dj_rb_")
    try:
        sf.write(os.path.join(tmp_dir, "in.wav"), audio.T, sr)
        subprocess.run(["rubberband", "-t", str(time_ratio),
                       os.path.join(tmp_dir, "in.wav"),
                       os.path.join(tmp_dir, "out.wav")],
                      capture_output=True, check=True)
        stretched, _ = sf.read(os.path.join(tmp_dir, "out.wav"), always_2d=True)
        return stretched.T
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def load_stems(track_path, target_sr=None):
    file_hash = get_file_hash(track_path)
    cache_dir = os.path.join(STEM_CACHE_DIR, file_hash)
    stem_names = ['drums', 'bass', 'vocals', 'other']
    if not (os.path.isdir(cache_dir) and all(
        os.path.exists(os.path.join(cache_dir, f"{s}.wav")) for s in stem_names)):
        raise RuntimeError(f"Stems not cached for {track_path} — run demucs first")
    stems = {}
    for name in stem_names:
        audio, file_sr = sf.read(os.path.join(cache_dir, f"{name}.wav"), always_2d=True)
        audio = audio.T
        if target_sr and file_sr != target_sr:
            audio = np.stack([librosa.resample(audio[ch], orig_sr=file_sr, target_sr=target_sr)
                              for ch in range(audio.shape[0])])
        stems[name] = audio
    return stems


def highpass(audio, sr, freq):
    if freq < 25: return audio
    sos = scipy_signal.butter(4, min(freq, sr * 0.45), btype='high', fs=sr, output='sos')
    return scipy_signal.sosfilt(sos, audio, axis=-1)


def make_envelope(keypoints, n_samples):
    env = np.zeros(n_samples)
    for i in range(len(keypoints) - 1):
        s0 = max(0, min(int(keypoints[i][0] * n_samples), n_samples))
        s1 = max(0, min(int(keypoints[i + 1][0] * n_samples), n_samples))
        if s1 > s0:
            env[s0:s1] = np.linspace(keypoints[i][1], keypoints[i + 1][1], s1 - s0)
    last_s = max(0, min(int(keypoints[-1][0] * n_samples), n_samples))
    env[last_s:] = keypoints[-1][1]
    return np.clip(env, 0.0, 1.0)


# ============================================================
#  REAL-TIME PLAYBACK ENGINE
# ============================================================
class DJEngine:
    """
    Real-time DJ engine. Pre-builds the entire mix buffer, then streams it.

    States:
      TRACK_A  — playing Track A normally
      LOOP     — looping Track A groove + mixing in Track B stems
      TRACK_B  — Track B takes over
    """

    def __init__(self, track_a_path, track_b_path):
        self.sr = 48000  # will be set from actual file
        self.position = 0
        self.playing = False
        self.mix_buffer = None

        self._prepare(track_a_path, track_b_path)

    def _prepare(self, track_a_path, track_b_path):
        """Pre-compute the full mix. This is the 'thinking' phase."""

        print("\n" + "=" * 60)
        print("  REAL-TIME DJ — PREPARING MIX")
        print("=" * 60)

        # --- Load audio ---
        print("\n  Loading tracks...")
        y_a, self.sr = sf.read(track_a_path, always_2d=True); y_a = y_a.T
        y_b, sr_b = sf.read(track_b_path, always_2d=True); y_b = y_b.T
        if sr_b != self.sr:
            y_b = np.stack([librosa.resample(y_b[ch], orig_sr=sr_b, target_sr=self.sr)
                            for ch in range(y_b.shape[0])])
        sr = self.sr

        # --- Beat analysis ---
        print("  Analyzing beats (madmom)...")
        beats_a = detect_beats(track_a_path)
        beats_b = detect_beats(track_b_path)
        bars_a = detect_downbeats(track_a_path)
        bars_b = detect_downbeats(track_b_path)
        bpm_a = compute_bpm(beats_a)
        bpm_b = compute_bpm(beats_b)
        lufs_a = measure_lufs(y_a, sr)

        print(f"  Track A: {bpm_a} BPM | Track B: {bpm_b} BPM")

        # --- Stems ---
        print("  Loading stems...")
        stems_a = load_stems(track_a_path, target_sr=sr)
        stems_b = load_stems(track_b_path, target_sr=sr)

        # --- Time stretch Track B ---
        rb_ratio = bpm_b / bpm_a
        if abs(rb_ratio - 1.0) > 0.002:
            print(f"  Stretching Track B: {bpm_b} → {bpm_a} BPM...")
            for name in stems_b:
                stems_b[name] = time_stretch(stems_b[name], sr, rb_ratio)
            bars_b = bars_b * rb_ratio

        # LUFS match
        min_len_b = min(s.shape[1] for s in stems_b.values())
        y_b_full = sum(s[:, :min_len_b] for s in stems_b.values())
        b_lufs = measure_lufs(y_b_full, sr)
        if not np.isinf(b_lufs):
            b_gain = 10 ** ((lufs_a - b_lufs) / 20.0)
            for name in stems_b:
                stems_b[name] = stems_b[name] * b_gain
            y_b_full = y_b_full * b_gain
            print(f"  LUFS matched (gain: {b_gain:.3f})")

        # --- Loop section: Track A bars 41-48 ---
        loop_start_idx = min(40, len(bars_a) - 2)
        loop_end_idx = min(48, len(bars_a) - 1)
        loop_start_s = float(bars_a[loop_start_idx])
        loop_end_s = float(bars_a[loop_end_idx])
        loop_start = int(loop_start_s * sr)
        loop_end = int(loop_end_s * sr)

        print(f"  Loop: bars {loop_start_idx+1}-{loop_end_idx} ({loop_start_s:.1f}-{loop_end_s:.1f}s)")

        # Build seamless loop (3x)
        n_repeats = 3
        xfade = min(int(0.03 * sr), (loop_end - loop_start) // 4)

        a_looped = {}
        for name in stems_a:
            end = min(loop_end, stems_a[name].shape[1])
            section = stems_a[name][:, loop_start:end]
            # Seamless loop
            n_ch, loop_len = section.shape
            xf = min(xfade, loop_len // 4)
            eff_len = loop_len - xf
            total_len = eff_len * n_repeats + xf
            result = np.zeros((n_ch, total_len))
            for rep in range(n_repeats):
                offset = rep * eff_len
                chunk = section.copy()
                if rep > 0:
                    chunk[:, :xf] *= np.linspace(0, 1, xf)[np.newaxis, :]
                if rep < n_repeats - 1:
                    chunk[:, -xf:] *= np.linspace(1, 0, xf)[np.newaxis, :]
                result[:, offset:offset + loop_len] += chunk
            a_looped[name] = result

        loop_n = min(s.shape[1] for s in a_looped.values())
        for name in a_looped:
            a_looped[name] = a_looped[name][:, :loop_n]

        # --- Track B cue at bar 25 ---
        b_cue_idx = min(24, len(bars_b) - 1)
        b_cue_s = float(bars_b[b_cue_idx])
        b_cue_sample = int(b_cue_s * sr)

        print(f"  Track B cue: bar {b_cue_idx+1} at {b_cue_s:.1f}s")

        b_section = {}
        for name in stems_b:
            end = min(b_cue_sample + loop_n, stems_b[name].shape[1])
            seg = stems_b[name][:, b_cue_sample:end]
            if seg.shape[1] < loop_n:
                pad = np.zeros((seg.shape[0], loop_n - seg.shape[1]))
                seg = np.concatenate([seg, pad], axis=1)
            b_section[name] = seg[:, :loop_n]

        # --- HP sweep on Track B drums (first 50%) ---
        hp_end = int(0.50 * loop_n)
        n_chunks = 32
        chunk_size = hp_end // n_chunks
        freqs = np.linspace(6000, 80, n_chunks)
        b_drums_copy = b_section['drums'][:, :hp_end].copy()
        for i in range(n_chunks):
            s0 = i * chunk_size
            s1 = s0 + chunk_size if i < n_chunks - 1 else hp_end
            if freqs[i] > 30:
                b_drums_copy[:, s0:s1] = highpass(b_drums_copy[:, s0:s1], sr, freqs[i])
        b_section['drums'][:, :hp_end] = b_drums_copy

        # --- Reverb wash on Track A 'other' ---
        rev_start = int(0.2 * loop_n)
        rev_end = int(0.75 * loop_n)
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=250),
            Reverb(room_size=0.85, damping=0.6, wet_level=0.4, dry_level=0.0),
            LowpassFilter(cutoff_frequency_hz=6000),
        ])
        a_other_rev = board(a_looped['other'][:, rev_start:rev_end].copy().astype(np.float32), sr)

        # --- Build envelopes ---
        n = loop_n
        swap_pct = 0.65

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

        # --- Mix the transition ---
        print("  Building transition...")
        transition = np.zeros((2, n))
        for stem in ['drums', 'bass', 'vocals', 'other']:
            a_contrib = a_looped[stem] * env_a[stem][np.newaxis, :]
            b_contrib = b_section[stem] * env_b[stem][np.newaxis, :]

            if stem == 'other':
                rev_len = min(a_other_rev.shape[1], rev_end - rev_start)
                rev_blend = np.linspace(0, 0.6, rev_len)
                dry_blend = np.linspace(1, 0.4, rev_len)
                a_contrib[:, rev_start:rev_start+rev_len] = (
                    a_contrib[:, rev_start:rev_start+rev_len] * dry_blend[np.newaxis, :] +
                    a_other_rev[:, :rev_len] * rev_blend[np.newaxis, :] *
                    env_a['other'][rev_start:rev_start+rev_len][np.newaxis, :]
                )

            transition += a_contrib + b_contrib

        # Soft clip
        trans_peak = np.max(np.abs(transition))
        if trans_peak > 0.95:
            scale = 1.2 / trans_peak
            transition = np.tanh(transition * scale) / np.tanh(scale)

        # --- Assemble full mix ---
        a_before = y_a[:, :loop_start]

        b_continue_from = b_cue_sample + loop_n
        b_after = y_b_full[:, b_continue_from:] if b_continue_from < y_b_full.shape[1] else np.zeros((2, sr))

        # Anti-click crossfades at junctions (5ms)
        xf_len = min(int(0.005 * sr), 512)
        if xf_len > 0 and a_before.shape[1] > xf_len and transition.shape[1] > xf_len:
            fo = np.linspace(1, 0, xf_len)[np.newaxis, :]
            fi = np.linspace(0, 1, xf_len)[np.newaxis, :]
            blend = a_before[:, -xf_len:] * fo + transition[:, :xf_len] * fi
            a_before = np.concatenate([a_before[:, :-xf_len], blend], axis=1)
            transition = transition[:, xf_len:]

        if xf_len > 0 and transition.shape[1] > xf_len and b_after.shape[1] > xf_len:
            fo = np.linspace(1, 0, xf_len)[np.newaxis, :]
            fi = np.linspace(0, 1, xf_len)[np.newaxis, :]
            blend = transition[:, -xf_len:] * fo + b_after[:, :xf_len] * fi
            transition = np.concatenate([transition[:, :-xf_len], blend], axis=1)
            b_after = b_after[:, xf_len:]

        self.mix_buffer = np.concatenate([a_before, transition, b_after], axis=1)
        self.mix_buffer = np.clip(self.mix_buffer, -0.98, 0.98)

        # Store markers for display
        self.markers = {
            'loop_start': a_before.shape[1],
            'loop_end': a_before.shape[1] + transition.shape[1],
            'swap': a_before.shape[1] + int(swap_pct * loop_n),
            'total': self.mix_buffer.shape[1],
        }

        dur = self.mix_buffer.shape[1] / sr
        print(f"\n  Mix ready: {dur:.1f}s ({dur/60:.1f} min)")
        print(f"  Transition at {a_before.shape[1]/sr:.1f}s, swap at {self.markers['swap']/sr:.1f}s")

    def play(self):
        """Stream the mix in real-time."""
        sr = self.sr
        total = self.mix_buffer.shape[1]
        dur = total / sr
        self.position = 0
        self.playing = True

        print("\n" + "=" * 60)
        print("  NOW PLAYING — CLAUDE'S DJ MIX")
        print("=" * 60)
        print(f"  Duration: {dur:.1f}s | SR: {sr}")
        print(f"  Press Ctrl+C to stop\n")

        block_size = 2048

        def callback(outdata, frames, time_info, status):
            if status:
                print(f"  ⚠ {status}")

            pos = self.position
            end = min(pos + frames, total)
            n_frames = end - pos

            if n_frames <= 0:
                outdata[:] = 0
                self.playing = False
                raise sd.CallbackStop()

            outdata[:n_frames] = self.mix_buffer[:, pos:end].T
            if n_frames < frames:
                outdata[n_frames:] = 0
                self.playing = False
                raise sd.CallbackStop()

            self.position = end

        stream = sd.OutputStream(
            samplerate=sr,
            channels=2,
            blocksize=block_size,
            callback=callback,
            dtype='float32',
        )

        try:
            with stream:
                while self.playing:
                    pos = self.position
                    t = pos / sr
                    pct = pos / total * 100

                    # Status line
                    state = "TRACK A"
                    if pos >= self.markers['loop_end']:
                        state = "TRACK B"
                    elif pos >= self.markers['swap']:
                        state = "TRACK B (bass dropped!)"
                    elif pos >= self.markers['loop_start']:
                        state = "TRANSITION (looping + mixing)"

                    bar_width = 40
                    filled = int(pct / 100 * bar_width)
                    bar = "█" * filled + "░" * (bar_width - filled)

                    # Mark the transition zone in the bar
                    trans_start_pct = self.markers['loop_start'] / total * 100
                    trans_end_pct = self.markers['loop_end'] / total * 100
                    swap_pct = self.markers['swap'] / total * 100

                    sys.stdout.write(
                        f"\r  {bar} {pct:5.1f}% | {t:5.1f}s | {state:<30}"
                    )
                    sys.stdout.flush()
                    time.sleep(0.1)

            print(f"\n\n  Playback finished.")

        except KeyboardInterrupt:
            print(f"\n\n  Stopped at {self.position/sr:.1f}s")


def main():
    track_a = sys.argv[1] if len(sys.argv) > 1 else "track1.mp3"
    track_b = sys.argv[2] if len(sys.argv) > 2 else "track2.mp3"

    engine = DJEngine(track_a, track_b)
    engine.play()


if __name__ == "__main__":
    main()
