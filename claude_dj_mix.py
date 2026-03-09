#!/usr/bin/env python3
"""
Claude's DJ Mix — Hand-crafted transition for track1.mp3 → track2.mp3

This is NOT a generic algorithm. This is a bespoke mix plan crafted by
analyzing both tracks' structure, stems, energy, and harmonic content.

Track A: G# major, 122.4 BPM — loop bars 41-48 (the groove, minimal vocals)
Track B: A major, 120.0 BPM — cue at bar 25 (second drop, pure instrumental)

Key clash (G#/A semitone): mitigated by avoiding melodic overlap.
Bass swap at 65% of transition, aligned to bar boundaries.
"""

import os
import sys
import hashlib
import subprocess
import tempfile
import shutil
import time as time_mod

import numpy as np
np.int = int
np.float = float
np.complex = complex

import librosa
import soundfile as sf
import madmom
import pyloudnorm
from scipy import signal as scipy_signal
from pedalboard import Pedalboard, Reverb, HighpassFilter, LowpassFilter

STEM_CACHE_DIR = "stem_cache"


# ============================================================
#  UTILITIES
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
    if len(beats) < 2:
        raise ValueError("Not enough beats")
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
        tmp_in = os.path.join(tmp_dir, "in.wav")
        tmp_out = os.path.join(tmp_dir, "out.wav")
        sf.write(tmp_in, audio.T, sr)
        cmd = ["rubberband", "-t", str(time_ratio), tmp_in, tmp_out]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rubberband failed: {result.stderr}")
        stretched, _ = sf.read(tmp_out, always_2d=True)
        return stretched.T
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def separate_stems(track_path, target_sr=None):
    file_hash = get_file_hash(track_path)
    cache_dir = os.path.join(STEM_CACHE_DIR, file_hash)
    stem_names = ['drums', 'bass', 'vocals', 'other']
    if os.path.isdir(cache_dir) and all(
        os.path.exists(os.path.join(cache_dir, f"{s}.wav")) for s in stem_names):
        print(f"  [cache] Stems: {os.path.basename(track_path)}")
        stems = {}
        for name in stem_names:
            audio, file_sr = sf.read(os.path.join(cache_dir, f"{name}.wav"), always_2d=True)
            audio = audio.T
            if target_sr and file_sr != target_sr:
                audio = np.stack([librosa.resample(audio[ch], orig_sr=file_sr, target_sr=target_sr)
                                  for ch in range(audio.shape[0])])
            stems[name] = audio
        return stems
    raise RuntimeError(f"Stems not cached for {track_path} — run demucs first")


# ============================================================
#  DSP TOOLS
# ============================================================
def make_envelope(keypoints, n_samples):
    """Build gain envelope from (pct, gain) keypoints."""
    env = np.zeros(n_samples)
    for i in range(len(keypoints) - 1):
        s0 = max(0, min(int(keypoints[i][0] * n_samples), n_samples))
        s1 = max(0, min(int(keypoints[i + 1][0] * n_samples), n_samples))
        if s1 > s0:
            env[s0:s1] = np.linspace(keypoints[i][1], keypoints[i + 1][1], s1 - s0)
    last_s = max(0, min(int(keypoints[-1][0] * n_samples), n_samples))
    env[last_s:] = keypoints[-1][1]
    return np.clip(env, 0.0, 1.0)


def make_seamless_loop(section, n_repeats, xfade_samples=2048):
    """Seamlessly loop with crossfade at boundaries."""
    n_ch, loop_len = section.shape
    xfade_samples = min(xfade_samples, loop_len // 4)
    effective_len = loop_len - xfade_samples
    total_len = effective_len * n_repeats + xfade_samples
    result = np.zeros((n_ch, total_len))
    for rep in range(n_repeats):
        offset = rep * effective_len
        chunk = section.copy()
        if rep > 0:
            chunk[:, :xfade_samples] *= np.linspace(0, 1, xfade_samples)[np.newaxis, :]
        if rep < n_repeats - 1:
            chunk[:, -xfade_samples:] *= np.linspace(1, 0, xfade_samples)[np.newaxis, :]
        result[:, offset:offset + loop_len] += chunk
    return result


def highpass(audio, sr, freq):
    if freq < 25: return audio
    sos = scipy_signal.butter(4, min(freq, sr * 0.45), btype='high', fs=sr, output='sos')
    return scipy_signal.sosfilt(sos, audio, axis=-1)


def lowpass(audio, sr, freq):
    if freq > sr * 0.45: return audio
    sos = scipy_signal.butter(4, max(freq, 25), btype='low', fs=sr, output='sos')
    return scipy_signal.sosfilt(sos, audio, axis=-1)


def reverb_wash(audio, sr, wet=0.35):
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=250),
        Reverb(room_size=0.85, damping=0.6, wet_level=wet, dry_level=0.0),
        LowpassFilter(cutoff_frequency_hz=6000),
    ])
    return board(audio.astype(np.float32), sr)


def hp_sweep(audio, sr, start_freq, end_freq):
    """Apply a high-pass filter that sweeps from start_freq to end_freq over time."""
    n_samples = audio.shape[1]
    # Process in chunks with varying HP frequency
    n_chunks = 32
    chunk_size = n_samples // n_chunks
    result = np.zeros_like(audio)
    freqs = np.linspace(start_freq, end_freq, n_chunks)

    for i in range(n_chunks):
        s0 = i * chunk_size
        s1 = s0 + chunk_size if i < n_chunks - 1 else n_samples
        freq = freqs[i]
        if freq < 30:
            result[:, s0:s1] = audio[:, s0:s1]
        else:
            result[:, s0:s1] = highpass(audio[:, s0:s1], sr, freq)

    return result


# ============================================================
#  THE MIX
# ============================================================
def mix():
    t0 = time_mod.time()
    track_a_path = "track1.mp3"
    track_b_path = "track2.mp3"
    output_path = "claude_dj_mix.wav"

    print("=" * 60)
    print("  CLAUDE'S DJ MIX")
    print("  track1.mp3 (G#, 122 BPM) → track2.mp3 (A, 120 BPM)")
    print("=" * 60)

    # ==========================================================
    # 1. LOAD & ANALYZE
    # ==========================================================
    print("\n[1/6] LOADING...")
    y_a, sr = sf.read(track_a_path, always_2d=True); y_a = y_a.T
    y_b, sr_b = sf.read(track_b_path, always_2d=True); y_b = y_b.T
    if sr_b != sr:
        y_b = np.stack([librosa.resample(y_b[ch], orig_sr=sr_b, target_sr=sr)
                        for ch in range(y_b.shape[0])])

    beats_a = detect_beats(track_a_path)
    bars_a = detect_downbeats(track_a_path)
    beats_b = detect_beats(track_b_path)
    bars_b = detect_downbeats(track_b_path)

    bpm_a = compute_bpm(beats_a)
    bpm_b = compute_bpm(beats_b)
    lufs_a = measure_lufs(y_a, sr)

    print(f"  Track A: {bpm_a} BPM, {lufs_a:.1f} LUFS, {y_a.shape[1]/sr:.1f}s")
    print(f"  Track B: {bpm_b} BPM, {y_b.shape[1]/sr:.1f}s")
    print(f"  Bars A: {len(bars_a)}, Bars B: {len(bars_b)}")

    # ==========================================================
    # 2. STEMS & TIME STRETCH
    # ==========================================================
    print("\n[2/6] STEMS & STRETCH...")
    stems_a = separate_stems(track_a_path, target_sr=sr)
    stems_b = separate_stems(track_b_path, target_sr=sr)

    # Stretch Track B to match Track A's BPM
    rb_ratio = bpm_b / bpm_a
    if abs(rb_ratio - 1.0) > 0.002:
        print(f"  Stretching Track B: {bpm_b} → {bpm_a} BPM (ratio {rb_ratio:.4f})")
        for name in stems_b:
            stems_b[name] = time_stretch(stems_b[name], sr, rb_ratio)
        bars_b = bars_b * rb_ratio
    else:
        print("  BPMs close enough — no stretch")

    # Reconstruct stretched Track B full audio
    min_len_b = min(s.shape[1] for s in stems_b.values())
    y_b_full = sum(s[:, :min_len_b] for s in stems_b.values())

    # LUFS match
    b_lufs = measure_lufs(y_b_full, sr)
    if not np.isinf(b_lufs):
        b_gain = 10 ** ((lufs_a - b_lufs) / 20.0)
        print(f"  LUFS match: {b_lufs:.1f} → {lufs_a:.1f} LUFS (gain: {b_gain:.3f})")
        for name in stems_b:
            stems_b[name] = stems_b[name] * b_gain
        y_b_full = y_b_full * b_gain

    # ==========================================================
    # 3. LOOP TRACK A (bars 41-48 = the groove)
    # ==========================================================
    print("\n[3/6] LOOPING TRACK A GROOVE...")

    # Bars 41-48 = indices 40-47 (0-indexed)
    # But bars_a might not have exactly that many. Use the analyzed positions.
    loop_start_idx = 40  # bar 41 (0-indexed)
    loop_end_idx = 48     # bar 49 (exclusive)

    # Clamp to available bars
    loop_start_idx = min(loop_start_idx, len(bars_a) - 2)
    loop_end_idx = min(loop_end_idx, len(bars_a) - 1)

    loop_start_s = float(bars_a[loop_start_idx])
    loop_end_s = float(bars_a[loop_end_idx])
    loop_start = int(loop_start_s * sr)
    loop_end = int(loop_end_s * sr)

    print(f"  Loop section: bars {loop_start_idx+1}-{loop_end_idx} ({loop_start_s:.1f}s - {loop_end_s:.1f}s)")

    # Verify energy
    loop_rms = np.sqrt(np.mean(y_a[:, loop_start:loop_end] ** 2))
    print(f"  Loop RMS: {loop_rms:.4f} (should be >0.1 = groove)")

    # Loop 3x = 24 bars
    n_repeats = 3
    xfade = min(int(0.03 * sr), (loop_end - loop_start) // 4)  # 30ms

    a_looped = {}
    for name in stems_a:
        end = min(loop_end, stems_a[name].shape[1])
        section = stems_a[name][:, loop_start:end]
        a_looped[name] = make_seamless_loop(section, n_repeats, xfade)

    loop_n = min(s.shape[1] for s in a_looped.values())
    for name in a_looped:
        a_looped[name] = a_looped[name][:, :loop_n]

    print(f"  Looped: {loop_n/sr:.1f}s ({n_repeats}x{loop_end_idx - loop_start_idx} bars = {n_repeats * (loop_end_idx - loop_start_idx)} bars)")

    # ==========================================================
    # 4. CUE TRACK B AT BAR 25 (second drop, instrumental)
    # ==========================================================
    print("\n[4/6] CUEING TRACK B...")

    # Bar 25 = index 24 (0-indexed) in original Track B bars
    b_cue_idx = 24  # bar 25
    b_cue_idx = min(b_cue_idx, len(bars_b) - 1)
    b_cue_s = float(bars_b[b_cue_idx])
    b_cue_sample = int(b_cue_s * sr)

    print(f"  Track B cue: bar {b_cue_idx+1} at {b_cue_s:.1f}s (second drop, pure instrumental)")

    # Extract Track B stems for the transition duration
    b_section = {}
    for name in stems_b:
        end = min(b_cue_sample + loop_n, stems_b[name].shape[1])
        seg = stems_b[name][:, b_cue_sample:end]
        if seg.shape[1] < loop_n:
            pad = np.zeros((seg.shape[0], loop_n - seg.shape[1]))
            seg = np.concatenate([seg, pad], axis=1)
        b_section[name] = seg[:, :loop_n]

    # Verify Track B section energy
    b_mix = sum(s for s in b_section.values())
    b_rms = np.sqrt(np.mean(b_mix ** 2))
    b_vox_rms = np.sqrt(np.mean(b_section['vocals'] ** 2))
    print(f"  Track B section RMS: {b_rms:.4f}, vocals RMS: {b_vox_rms:.4f}")

    # ==========================================================
    # 5. THE TRANSITION (the magic)
    # ==========================================================
    print("\n[5/6] MIXING TRANSITION...")
    n = loop_n

    # --- PHASE BOUNDARIES ---
    # Phase 1: 0-25%   — Ghost hi-hats from B
    # Phase 2: 25-50%  — B drums opening up, A melody dissolving
    # Phase 3: 50-65%  — Dual drums, tension build
    # Phase 4: 65%     — BASS SWAP
    # Phase 5: 65-100% — Track B full takeover

    swap_pct = 0.65

    # --- TRACK A ENVELOPES ---
    env_a = {
        # Drums: strong groove, gradually hand off to B
        'drums': make_envelope([
            (0, 1.0),        # Full groove
            (0.25, 1.0),     # Still full through phase 1
            (0.4, 0.8),      # Start pulling back
            (swap_pct, 0.0), # Gone at bass swap
        ], n),

        # Bass: holds strong until the swap, then INSTANT CUT
        'bass': make_envelope([
            (0, 1.0),
            (swap_pct - 0.01, 1.0),  # Hold till the last moment
            (swap_pct, 0.0),          # Instant cut
        ], n),

        # Vocals: fade early — avoid clash with Track B (key mismatch)
        # Track A bars 41-48 have low vocals (0.09 RMS) so this is mostly precautionary
        'vocals': make_envelope([
            (0, 0.8),       # Already reduced (this section has light vocals)
            (0.15, 0.5),    # Quick fade
            (0.3, 0.0),     # Gone by 30% — well before any B content
        ], n),

        # Other (melodic/harmonic): fade with reverb to blur the G# tonality
        'other': make_envelope([
            (0, 1.0),
            (0.2, 0.8),     # Start fading
            (0.4, 0.4),     # Half
            (0.6, 0.15),    # Almost gone
            (0.75, 0.0),    # Fully gone
        ], n),
    }

    # --- TRACK B ENVELOPES ---
    env_b = {
        # Drums: ghost hi-hats → open up → full
        'drums': make_envelope([
            (0, 0.0),        # Silent
            (0.05, 0.12),    # Ghost hi-hats appear
            (0.15, 0.2),     # Subtle presence
            (0.3, 0.35),     # Opening up
            (0.5, 0.6),      # Strong
            (swap_pct, 0.85), # Nearly full at swap
            (0.75, 1.0),     # Full
        ], n),

        # Bass: SILENT until swap, then INSTANT ENTRY
        'bass': make_envelope([
            (0, 0.0),
            (swap_pct - 0.01, 0.0),  # Dead silent
            (swap_pct, 1.0),          # BOOM — instant bass swap
        ], n),

        # Vocals: Track B bars 25-32 have low vocals (0.083 RMS)
        # Keep them suppressed during transition, let them in at the end
        'vocals': make_envelope([
            (0, 0.0),
            (0.6, 0.0),      # No vocals during overlap
            (swap_pct, 0.0),  # Still none at swap
            (0.75, 0.2),     # Gentle entry
            (0.9, 0.8),      # Building
            (1.0, 1.0),      # Full
        ], n),

        # Other (melodic): bring in AFTER Track A's melody is gone
        # Avoids G# vs A harmonic clash
        'other': make_envelope([
            (0, 0.0),
            (0.3, 0.0),      # Nothing while A's melody is present
            (0.4, 0.1),      # Tiny hint
            (0.55, 0.3),     # Building (A's melody at 0.15 here)
            (swap_pct, 0.6), # Established
            (0.8, 0.9),      # Nearly full
            (0.9, 1.0),      # Full
        ], n),
    }

    print(f"  Bass swap at {swap_pct*100:.0f}% = {swap_pct * n / sr:.1f}s into transition")

    # --- HP SWEEP on Track B drums ---
    # Phase 1 (0-25%): only hi-hats >6kHz
    # Phase 2 (25-50%): sweep down to let in more drums
    # Phase 3 (50%+): full spectrum
    hp_end_point = int(0.50 * n)
    b_drums_early = b_section['drums'][:, :hp_end_point].copy()
    b_drums_early = hp_sweep(b_drums_early, sr, 6000, 80)  # 6kHz → 80Hz sweep
    b_section['drums'][:, :hp_end_point] = b_drums_early
    print("  Track B drums: HP sweep 6kHz→80Hz over first 50%")

    # --- REVERB WASH on Track A 'other' stem ---
    # Blur the harmonic content as it fades
    reverb_start = int(0.2 * n)
    reverb_end = int(0.75 * n)
    a_other_section = a_looped['other'][:, reverb_start:reverb_end].copy()
    a_reverb = reverb_wash(a_other_section, sr, wet=0.4)
    print("  Track A melody: reverb wash (20-75%) to blur G# tonality")

    # --- ASSEMBLE THE TRANSITION ---
    transition = np.zeros((2, n))

    for stem in ['drums', 'bass', 'vocals', 'other']:
        a_contrib = a_looped[stem] * env_a[stem][np.newaxis, :]
        b_contrib = b_section[stem] * env_b[stem][np.newaxis, :]

        # For 'other' stem of Track A, blend in the reverb version
        if stem == 'other':
            rev_len = min(a_reverb.shape[1], reverb_end - reverb_start)
            # Create a reverb blend envelope (dry → wet as it fades)
            rev_blend = np.linspace(0, 0.6, rev_len)
            dry_blend = np.linspace(1, 0.4, rev_len)
            a_contrib[:, reverb_start:reverb_start + rev_len] = (
                a_contrib[:, reverb_start:reverb_start + rev_len] * dry_blend[np.newaxis, :] +
                a_reverb[:, :rev_len] * rev_blend[np.newaxis, :] * env_a['other'][reverb_start:reverb_start + rev_len][np.newaxis, :]
            )

        transition += a_contrib + b_contrib

    # --- SOFT CLIP (tanh) ---
    trans_peak = np.max(np.abs(transition))
    if trans_peak > 0.95:
        scale = 1.2 / trans_peak
        transition = np.tanh(transition * scale) / np.tanh(scale)
        new_peak = np.max(np.abs(transition))
        print(f"  Soft-clipped: {trans_peak:.3f} → {new_peak:.3f}")
    else:
        print(f"  Transition peak: {trans_peak:.3f} — clean")

    # ==========================================================
    # 6. ASSEMBLE FINAL MIX
    # ==========================================================
    print("\n[6/6] ASSEMBLING...")

    # Track A: everything before the loop point
    a_before = y_a[:, :loop_start]

    # Track B: continue from where transition left off
    b_continue_from = b_cue_sample + loop_n
    if b_continue_from < y_b_full.shape[1]:
        b_after = y_b_full[:, b_continue_from:]
    else:
        b_after = np.zeros((2, int(sr)))

    # 5ms anti-click crossfade at junctions
    xf_len = min(int(0.005 * sr), 512)

    # Junction 1: Track A → Transition
    if xf_len > 0 and a_before.shape[1] > xf_len and transition.shape[1] > xf_len:
        fade_out = np.linspace(1, 0, xf_len)[np.newaxis, :]
        fade_in = np.linspace(0, 1, xf_len)[np.newaxis, :]
        overlap_a = a_before[:, -xf_len:]
        overlap_t = transition[:, :xf_len]
        blend = overlap_a * fade_out + overlap_t * fade_in
        a_before = np.concatenate([a_before[:, :-xf_len], blend], axis=1)
        transition = transition[:, xf_len:]

    # Junction 2: Transition → Track B
    if xf_len > 0 and transition.shape[1] > xf_len and b_after.shape[1] > xf_len:
        fade_out = np.linspace(1, 0, xf_len)[np.newaxis, :]
        fade_in = np.linspace(0, 1, xf_len)[np.newaxis, :]
        overlap_t = transition[:, -xf_len:]
        overlap_b = b_after[:, :xf_len]
        blend = overlap_t * fade_out + overlap_b * fade_in
        transition = np.concatenate([transition[:, :-xf_len], blend], axis=1)
        b_after = b_after[:, xf_len:]

    final = np.concatenate([a_before, transition, b_after], axis=1)

    # Hard clip at ±0.98
    peak = np.max(np.abs(final))
    if peak > 0.98:
        n_clipped = np.sum(np.abs(final) > 0.98)
        total = final.shape[0] * final.shape[1]
        final = np.clip(final, -0.98, 0.98)
        print(f"  Hard-clipped: {n_clipped}/{total} samples ({n_clipped/total*100:.2f}%) at peak {peak:.3f}")
    else:
        print(f"  Peak: {peak:.3f} — clean, no clipping needed")

    sf.write(output_path, final.T, sr)

    # --- SUMMARY ---
    elapsed = time_mod.time() - t0
    dur = final.shape[1] / sr
    trans_dur = loop_n / sr

    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.0f}s → {output_path}")
    print(f"  Total: {dur/60:.1f} min ({dur:.1f}s)")
    print(f"  Track A solo: {a_before.shape[1]/sr:.1f}s")
    print(f"  Transition: {trans_dur:.1f}s (24 bars)")
    print(f"    - Loop: bars 41-48 x3 ({loop_start_s:.1f}-{loop_end_s:.1f}s)")
    print(f"    - Track B cued at bar 25 ({b_cue_s:.1f}s)")
    print(f"    - Bass swap at {swap_pct*100:.0f}% ({swap_pct*trans_dur:.1f}s)")
    print(f"  Track B solo: {b_after.shape[1]/sr:.1f}s")
    print(f"  Peak: {np.max(np.abs(final)):.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mix()
