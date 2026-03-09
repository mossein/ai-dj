#!/usr/bin/env python3
"""
AI DJ v3 — Stem-Based CDJ Technique

Does what real DJs do on CDJs + 3-band mixer:
1. Find a loopable section in Track A (breakdown/low-energy)
2. LOOP it to extend the runway
3. Layer in Track B stems one by one (hats -> perc -> synths -> bass swap)
4. Release loop, Track B takes over

Stems are used for intelligent layering, not just volume fades.
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
#  BEAT & DOWNBEAT DETECTION
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


# ============================================================
#  KEY DETECTION (Krumhansl-Kessler + Camelot)
# ============================================================
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT = {
    'C': '8B', 'Am': '8A', 'G': '9B', 'Em': '9A', 'D': '10B', 'Bm': '10A',
    'A': '11B', 'F#m': '11A', 'E': '12B', 'C#m': '12A', 'B': '1B', 'G#m': '1A',
    'F#': '2B', 'D#m': '2A', 'C#': '3B', 'A#m': '3A', 'G#': '4B', 'Fm': '4A',
    'D#': '5B', 'Cm': '5A', 'A#': '6B', 'Gm': '6A', 'F': '7B', 'Dm': '7A',
}

def detect_key(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    best_corr, best_key, best_mode = -1, 'C', 'major'
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        for profile, mode in [(MAJOR_PROFILE, 'major'), (MINOR_PROFILE, 'minor')]:
            corr = np.corrcoef(rolled, profile)[0, 1]
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, KEY_NAMES[shift], mode
    key_str = best_key if best_mode == 'major' else f'{best_key}m'
    return key_str, CAMELOT.get(key_str, '?')

def keys_compatible(cam_a, cam_b):
    if '?' in (cam_a, cam_b): return True
    num_a, mode_a = int(cam_a[:-1]), cam_a[-1]
    num_b, mode_b = int(cam_b[:-1]), cam_b[-1]
    if cam_a == cam_b: return True
    if mode_a == mode_b:
        diff = abs(num_a - num_b)
        if diff <= 1 or diff == 11: return True
    if num_a == num_b and mode_a != mode_b: return True
    return False


# ============================================================
#  ENERGY & STRUCTURE
# ============================================================
def analyze_energy(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / sr
    hop = sr // 2
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)

    valleys = []
    window = 16
    for i in range(window, len(rms_norm) - window):
        local_before = np.mean(rms_norm[max(0, i - window):i])
        if local_before > 0.4 and rms_norm[i] < local_before * 0.6:
            pct = (i * 0.5) / duration
            if 0.35 < pct < 0.85:
                valleys.append({'time_s': i * 0.5, 'pct': pct,
                                'energy_drop': float(local_before - rms_norm[i])})

    filtered = []
    for v in sorted(valleys, key=lambda x: -x['energy_drop']):
        if not any(abs(v['time_s'] - f['time_s']) < 10 for f in filtered):
            filtered.append(v)

    return {'duration_s': duration, 'rms': rms_norm,
            'valleys': sorted(filtered, key=lambda x: x['time_s'])}


def find_nearest_bar(time_s, bar_starts):
    if len(bar_starts) == 0: return time_s
    return float(bar_starts[np.argmin(np.abs(bar_starts - time_s))])


def find_mix_out_point(energy_data, bar_starts):
    valleys = energy_data['valleys']
    if not valleys:
        return find_nearest_bar(energy_data['duration_s'] * 0.65, bar_starts)
    ideal = [v for v in valleys if 0.45 < v['pct'] < 0.80]
    best = max(ideal or valleys, key=lambda v: v['energy_drop'])
    return find_nearest_bar(best['time_s'], bar_starts)


def find_track_b_entry(audio_path, bars_b, sr):
    """Find the best cue point for Track B — where the first drop/groove starts.

    Returns (intro_start_s, drop_s):
      - intro_start_s: where to start playing Track B (beginning of buildup)
      - drop_s: where the drop hits (bass + drums come in hard)

    We want to align the drop with the bass swap in the transition.
    """
    y, load_sr = librosa.load(audio_path, sr=22050)
    duration = len(y) / load_sr

    # Compute RMS energy with 0.5s windows
    hop = load_sr // 2
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)

    # Find the first big energy jump (the "drop")
    # Look for a sustained jump from low to high energy
    drop_time = None
    window = 8  # 4 seconds of context

    for i in range(window, len(rms_norm) - window):
        time_s = i * 0.5
        # Must be at least 15s into the track
        if time_s < 15:
            continue
        # Must be in the first 60% of the track
        if time_s / duration > 0.6:
            break

        before = np.mean(rms_norm[max(0, i - window):i])
        after = np.mean(rms_norm[i:i + window])

        # Big energy jump: before is quiet, after is loud
        if after > before * 1.8 and after > 0.5 and (after - before) > 0.2:
            drop_time = time_s
            break

    if drop_time is None:
        # Fallback: find first moment of sustained high energy
        for i in range(len(rms_norm)):
            time_s = i * 0.5
            if time_s < 10:
                continue
            # 4 seconds of high energy
            chunk = rms_norm[i:i + 8]
            if len(chunk) >= 8 and np.mean(chunk) > 0.6:
                drop_time = time_s
                break

    if drop_time is None:
        # Last resort: 30% into the track
        drop_time = duration * 0.3

    # Snap drop to nearest bar
    drop_s = find_nearest_bar(drop_time, bars_b)

    # The intro/buildup typically starts 16-32 bars before the drop
    # We want enough buildup to layer in during the transition
    bar_dur = np.median(np.diff(bars_b)) if len(bars_b) > 1 else 2.0
    buildup_bars = 16  # 16 bars of buildup
    intro_start_s = max(0, drop_s - buildup_bars * bar_dur)
    intro_start_s = find_nearest_bar(intro_start_s, bars_b)

    return intro_start_s, drop_s


# ============================================================
#  LOUDNESS
# ============================================================
def measure_lufs(audio, sr):
    meter = pyloudnorm.Meter(sr)
    if audio.ndim == 1: audio = np.stack([audio, audio])
    return meter.integrated_loudness(audio.T)

def match_loudness(audio, sr, target_lufs):
    current = measure_lufs(audio, sr)
    if np.isinf(current): return audio
    return audio * 10 ** ((target_lufs - current) / 20.0)


# ============================================================
#  TIME STRETCHING
# ============================================================
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


# ============================================================
#  STEM SEPARATION
# ============================================================
def get_file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()

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

    tmp_dir = tempfile.mkdtemp(prefix="dj_stems_")
    track_name = os.path.splitext(os.path.basename(track_path))[0]
    print(f"  [demucs] Separating: {os.path.basename(track_path)}...")
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs",
           "--out", tmp_dir, os.path.abspath(track_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed: {result.stderr}")

    stem_dir = os.path.join(tmp_dir, "htdemucs", track_name)
    stems = {}
    os.makedirs(cache_dir, exist_ok=True)
    for name in stem_names:
        src = os.path.join(stem_dir, f"{name}.wav")
        shutil.copy2(src, os.path.join(cache_dir, f"{name}.wav"))
        audio, file_sr = sf.read(src, always_2d=True)
        audio = audio.T
        if target_sr and file_sr != target_sr:
            audio = np.stack([librosa.resample(audio[ch], orig_sr=file_sr, target_sr=target_sr)
                              for ch in range(audio.shape[0])])
        stems[name] = audio
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return stems


# ============================================================
#  LOOP ENGINE
# ============================================================
def make_seamless_loop(section, n_repeats, xfade_samples=2048):
    """Seamlessly loop a musical section with crossfade at boundaries."""
    n_ch, loop_len = section.shape
    xfade_samples = min(xfade_samples, loop_len // 4)

    # Each repetition after the first overlaps by xfade_samples
    effective_len = loop_len - xfade_samples
    total_len = effective_len * n_repeats + xfade_samples
    result = np.zeros((n_ch, total_len))

    for rep in range(n_repeats):
        offset = rep * effective_len
        chunk = section.copy()

        # Fade in at start (except first rep)
        if rep > 0:
            chunk[:, :xfade_samples] *= np.linspace(0, 1, xfade_samples)[np.newaxis, :]
        # Fade out at end (except last rep)
        if rep < n_repeats - 1:
            chunk[:, -xfade_samples:] *= np.linspace(1, 0, xfade_samples)[np.newaxis, :]

        result[:, offset:offset + loop_len] += chunk

    return result


def loop_stems(stems, start_sample, end_sample, n_repeats, xfade_samples=2048):
    """Loop all stems of a track."""
    looped = {}
    for name, audio in stems.items():
        end = min(end_sample, audio.shape[1])
        section = audio[:, start_sample:end]
        looped[name] = make_seamless_loop(section, n_repeats, xfade_samples)
    return looped


# ============================================================
#  ENVELOPE BUILDER
# ============================================================
def make_envelope(keypoints, n_samples):
    """Build a gain envelope from (pct, gain) keypoints.
    pct is 0.0-1.0, gain is 0.0-1.0."""
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
#  EFFECTS
# ============================================================
def highpass(audio, sr, freq):
    if freq < 25: return audio
    sos = scipy_signal.butter(4, min(freq, sr * 0.45), btype='high', fs=sr, output='sos')
    return scipy_signal.sosfilt(sos, audio, axis=-1)


def reverb_wash(audio, sr, wet=0.35):
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=250),
        Reverb(room_size=0.85, damping=0.6, wet_level=wet, dry_level=0.0),
        LowpassFilter(cutoff_frequency_hz=6000),
    ])
    return board(audio.astype(np.float32), sr)


# ============================================================
#  THE MIXER
# ============================================================
def mix_tracks(track_a_path, track_b_path, output_path="ai_dj_v3_mix.wav"):
    t0 = time_mod.time()
    print("=" * 60)
    print("  AI DJ v3 — CDJ Stem Technique")
    print("=" * 60)

    # ==========================================================
    # 1. ANALYZE
    # ==========================================================
    print("\n[1/7] ANALYZING...")

    y_a, sr = sf.read(track_a_path, always_2d=True); y_a = y_a.T
    y_b, sr_b = sf.read(track_b_path, always_2d=True); y_b = y_b.T
    if sr_b != sr:
        y_b = np.stack([librosa.resample(y_b[ch], orig_sr=sr_b, target_sr=sr)
                        for ch in range(y_b.shape[0])])

    print("  Beats & downbeats (madmom RNN)...")
    beats_a = detect_beats(track_a_path)
    beats_b = detect_beats(track_b_path)
    bpm_a = compute_bpm(beats_a)
    bpm_b = compute_bpm(beats_b)
    bars_a = detect_downbeats(track_a_path)
    bars_b = detect_downbeats(track_b_path)

    key_a, cam_a = detect_key(track_a_path)
    key_b, cam_b = detect_key(track_b_path)
    harmonic = keys_compatible(cam_a, cam_b)

    energy_a = analyze_energy(track_a_path)
    lufs_a = measure_lufs(y_a, sr)
    lufs_b = measure_lufs(y_b, sr)

    print(f"\n  Track A: {key_a} ({cam_a}) @ {bpm_a} BPM | {y_a.shape[1]/sr:.1f}s | {lufs_a:.1f} LUFS")
    print(f"  Track B: {key_b} ({cam_b}) @ {bpm_b} BPM | {y_b.shape[1]/sr:.1f}s | {lufs_b:.1f} LUFS")
    print(f"  Harmonic: {'YES' if harmonic else 'CLASH'}")
    dips = [f"{v['time_s']:.0f}s" for v in energy_a['valleys']]
    print(f"  Energy dips (A): {dips or 'none'}")

    # ==========================================================
    # 2. SEPARATE STEMS
    # ==========================================================
    print("\n[2/7] SEPARATING STEMS...")
    stems_a = separate_stems(track_a_path, target_sr=sr)
    stems_b = separate_stems(track_b_path, target_sr=sr)

    # ==========================================================
    # 3. TIME STRETCH Track B STEMS
    # ==========================================================
    print("\n[3/7] TIME STRETCHING...")
    rb_ratio = bpm_b / bpm_a  # time_ratio for rubberband
    if abs(rb_ratio - 1.0) > 0.002:
        print(f"  Stretching Track B stems: {bpm_b} -> {bpm_a} BPM")
        for name in stems_b:
            stems_b[name] = time_stretch(stems_b[name], sr, rb_ratio)
            print(f"    B.{name}: {stems_b[name].shape[1]/sr:.1f}s")
        bars_b = bars_b * rb_ratio
    else:
        print("  BPMs match — no stretch")

    # Reconstruct full Track B from stretched stems and LUFS match
    min_stem_len = min(s.shape[1] for s in stems_b.values())
    y_b_stretched = sum(s[:, :min_stem_len] for s in stems_b.values())

    # LUFS match — apply gain to get Track B at same perceived loudness as Track A
    b_lufs = measure_lufs(y_b_stretched, sr)
    if not np.isinf(b_lufs):
        b_gain = 10 ** ((lufs_a - b_lufs) / 20.0)
        print(f"  LUFS gain for Track B: {b_gain:.3f} ({b_lufs:.1f} -> {lufs_a:.1f} LUFS)")
        for name in stems_b:
            stems_b[name] = stems_b[name] * b_gain
        y_b_stretched = y_b_stretched * b_gain
        pk_b = np.max(np.abs(y_b_stretched))
        print(f"  Track B peak after LUFS match: {pk_b:.3f}")
    # Don't peak-limit Track B here — we'll hard-clip the final output at ±0.98
    # A few clipped samples are inaudible. Reducing the whole track by 5dB is not.

    # ==========================================================
    # 4. FIND LOOP SECTION IN TRACK A
    # ==========================================================
    print("\n[4/7] FINDING LOOP POINT...")
    bar_dur = 4 * 60.0 / bpm_a
    bar_samples = int(bar_dur * sr)

    mix_out_s = find_mix_out_point(energy_a, bars_a)
    mix_out_bar_idx = np.argmin(np.abs(bars_a - mix_out_s)) if len(bars_a) > 0 else 0

    # Loop the 8 bars BEFORE the breakdown — that's the groove!
    # The mix-out point is where energy DROPS. We want the bars leading up to it.
    loop_bars = 8
    loop_end_bar_idx = mix_out_bar_idx  # Loop ENDS at the breakdown
    loop_start_bar_idx = max(0, loop_end_bar_idx - loop_bars)

    if loop_start_bar_idx < len(bars_a):
        loop_start_s = float(bars_a[loop_start_bar_idx])
    else:
        loop_start_s = max(0, mix_out_s - loop_bars * bar_dur)

    if loop_end_bar_idx < len(bars_a):
        loop_end_s = float(bars_a[loop_end_bar_idx])
    else:
        loop_end_s = mix_out_s

    loop_start = int(loop_start_s * sr)
    loop_end = int(loop_end_s * sr)

    # Verify we got the groove, not the breakdown
    loop_section_full = y_a[:, loop_start:loop_end]
    loop_rms = np.sqrt(np.mean(loop_section_full ** 2))
    print(f"  Loop section RMS: {loop_rms:.4f}")

    # How many times to repeat the loop (3x = 24 bars total)
    n_repeats = 3
    print(f"  Loop: {loop_start_s:.1f}s - {loop_end_s:.1f}s ({loop_bars} bars x{n_repeats} = {loop_bars * n_repeats} bars)")
    print(f"  (This is the groove BEFORE the energy dip at {mix_out_s:.1f}s)")

    # ==========================================================
    # 5. BUILD LOOPED TRACK A STEMS
    # ==========================================================
    print("\n[5/7] BUILDING LOOP...")
    xfade = min(int(0.03 * sr), bar_samples // 4)  # 30ms crossfade
    a_looped = loop_stems(stems_a, loop_start, loop_end, n_repeats, xfade)
    loop_n = min(s.shape[1] for s in a_looped.values())
    for name in a_looped:
        a_looped[name] = a_looped[name][:, :loop_n]

    print(f"  Looped region: {loop_n/sr:.1f}s ({loop_n} samples)")

    # ==========================================================
    # 6. STEM TRANSITION (the magic)
    # ==========================================================
    print("\n[6/7] STEM MIXING...")

    # Track B: find smart entry point (intro + drop)
    print("  Finding Track B entry point...")
    b_intro_s, b_drop_s = find_track_b_entry(track_b_path, bars_b, sr)
    # After time-stretching, bars_b was already adjusted
    print(f"  Track B: intro at {b_intro_s:.1f}s, drop at {b_drop_s:.1f}s")

    # Align Track B's drop with the bass swap in the transition
    # The bass swap should happen when Track B's drop hits
    b_intro_start = int(b_intro_s * sr)
    b_drop_sample = int(b_drop_s * sr)

    # How far into Track B (from intro_start) the drop occurs
    drop_offset = b_drop_sample - b_intro_start
    # The swap should be at this fraction of the transition
    swap_pct = np.clip(drop_offset / loop_n, 0.4, 0.85) if loop_n > 0 else 0.65
    print(f"  Bass swap at {swap_pct*100:.0f}% (aligned with Track B drop)")

    b_section = {}
    for name in stems_b:
        end = min(b_intro_start + loop_n, stems_b[name].shape[1])
        seg = stems_b[name][:, b_intro_start:end]
        # Pad if shorter than loop
        if seg.shape[1] < loop_n:
            pad = np.zeros((seg.shape[0], loop_n - seg.shape[1]))
            seg = np.concatenate([seg, pad], axis=1)
        b_section[name] = seg[:, :loop_n]

    n = loop_n

    # --- VOCAL COLLISION DETECTION ---
    # Check where each track has active vocals and prevent overlap
    def stem_energy_profile(stem_audio, hop_samples=None):
        """Get per-bar energy of a stem."""
        if hop_samples is None:
            hop_samples = n // 20  # 20 segments
        n_segs = max(1, stem_audio.shape[1] // hop_samples)
        energies = np.zeros(n_segs)
        for i in range(n_segs):
            chunk = stem_audio[:, i * hop_samples:(i + 1) * hop_samples]
            energies[i] = np.sqrt(np.mean(chunk ** 2))
        return energies

    voc_a_energy = stem_energy_profile(a_looped['vocals'])
    voc_b_energy = stem_energy_profile(b_section['vocals'])
    voc_a_active = voc_a_energy > 0.01  # threshold for "vocals present"
    voc_b_active = voc_b_energy > 0.01

    # Find where Track B vocals first appear
    b_vocal_start_pct = 1.0
    for i, active in enumerate(voc_b_active):
        if active:
            b_vocal_start_pct = i / len(voc_b_active)
            break

    # Track A vocals must be fully out before Track B vocals start
    a_vocal_end_pct = min(swap_pct - 0.1, b_vocal_start_pct - 0.05)
    a_vocal_end_pct = max(0.15, a_vocal_end_pct)  # at least 15% for Track A vocals

    # Track B vocals: don't bring in until Track A vocals are gone
    b_vocal_in_pct = max(a_vocal_end_pct + 0.05, swap_pct)

    print(f"  Vocal collision prevention: A vocals out by {a_vocal_end_pct*100:.0f}%, B vocals in at {b_vocal_in_pct*100:.0f}%")

    # Track A envelopes (fading out):
    env_a = {
        'drums':  make_envelope([(0, 1.0), (0.4, 0.7), (swap_pct, 0.0)], n),
        'bass':   make_envelope([(0, 1.0), (swap_pct - 0.02, 1.0), (swap_pct, 0.0)], n),  # sudden cut
        'vocals': make_envelope([(0, 1.0), (a_vocal_end_pct - 0.15, 0.5), (a_vocal_end_pct, 0.0)], n),
        'other':  make_envelope([(0, 1.0), (0.5, 0.5), (0.8, 0.1), (1.0, 0.0)], n),
    }

    # Track B envelopes (layering in):
    env_b = {
        'drums':  make_envelope([(0, 0.0), (0.1, 0.15), (0.35, 0.4), (swap_pct, 0.8), (0.8, 1.0)], n),
        'bass':   make_envelope([(0, 0.0), (swap_pct - 0.02, 0.0), (swap_pct, 1.0)], n),  # sudden entry
        'vocals': make_envelope([(0, 0.0), (b_vocal_in_pct, 0.0), (b_vocal_in_pct + 0.1, 0.5), (0.9, 1.0)], n),
        'other':  make_envelope([(0, 0.0), (0.15, 0.1), (0.4, 0.3), (swap_pct, 0.7), (0.85, 1.0)], n),
    }

    print(f"  Bass swap at {swap_pct*100:.0f}% ({swap_pct * loop_n / sr:.1f}s into transition)")

    # HP filter Track B drums in early phase (just hi-hats/rides)
    # Below 35%, only let through frequencies above 4kHz
    hp_cutoff_point = int(0.35 * n)
    b_drums_early = b_section['drums'][:, :hp_cutoff_point].copy()
    b_drums_early = highpass(b_drums_early, sr, 4000)
    b_section['drums'][:, :hp_cutoff_point] = b_drums_early
    print("  Track B drums HP filtered (>4kHz) in first 35% — just hi-hats")

    # Reverb wash on Track A's other/vocals as they fade out
    reverb_start = int(0.4 * n)
    a_other_tail = a_looped['other'][:, reverb_start:].copy()
    a_reverb = reverb_wash(a_other_tail, sr, wet=0.3)

    # --- MIX THE TRANSITION ---
    transition = np.zeros((2, n))

    for stem in ['drums', 'bass', 'vocals', 'other']:
        transition += a_looped[stem] * env_a[stem][np.newaxis, :]
        transition += b_section[stem] * env_b[stem][np.newaxis, :]

    # Blend reverb tail
    rev_len = min(a_reverb.shape[1], n - reverb_start)
    if rev_len > 0:
        rev_env = np.linspace(0, 0.2, rev_len)
        transition[:, reverb_start:reverb_start + rev_len] += a_reverb[:, :rev_len] * rev_env[np.newaxis, :]

    # Soft-clip transition to tame peaks without crushing overall level
    # tanh-based soft clipper: preserves quieter parts, gently limits peaks
    trans_peak = np.max(np.abs(transition))
    if trans_peak > 0.95:
        # Scale so that 0.95 maps to tanh threshold, then soft clip
        scale = 1.2 / trans_peak  # map peak to ~1.2 (tanh(1.2) ≈ 0.83)
        transition = np.tanh(transition * scale) / np.tanh(scale)
        new_peak = np.max(np.abs(transition))
        print(f"  Transition soft-clipped: {trans_peak:.3f} -> {new_peak:.3f}")

    # ==========================================================
    # 7. ASSEMBLE
    # ==========================================================
    print("\n[7/7] ASSEMBLING...")

    # Track A: everything before the loop
    a_before = y_a[:, :loop_start]

    # Track B: continue from where the transition left off
    # The transition used Track B from b_intro_start to b_intro_start + loop_n
    b_continue_from = b_intro_start + loop_n
    if b_continue_from < y_b_stretched.shape[1]:
        b_after = y_b_stretched[:, b_continue_from:].copy()
    else:
        b_after = np.zeros((2, int(sr)))

    # Tiny crossfade at transition -> Track B junction (5ms, just anti-click)
    xf_len = min(int(0.005 * sr), 512)
    if xf_len > 0 and transition.shape[1] > xf_len and b_after.shape[1] > xf_len:
        # Overlap-add crossfade (no energy dip)
        overlap_a = transition[:, -xf_len:]
        overlap_b = b_after[:, :xf_len]
        blend = overlap_a * np.linspace(1, 0, xf_len)[np.newaxis, :] + \
                overlap_b * np.linspace(0, 1, xf_len)[np.newaxis, :]
        transition[:, -xf_len:] = blend
        b_after = b_after[:, xf_len:]

    final = np.concatenate([a_before, transition, b_after], axis=1)

    # Hard-clip at ±0.98 (a few clipped samples are inaudible;
    # reducing the whole mix by 5dB is very audible)
    peak = np.max(np.abs(final))
    if peak > 0.98:
        n_clipped = np.sum(np.abs(final) > 0.98)
        total = final.shape[0] * final.shape[1]
        final = np.clip(final, -0.98, 0.98)
        print(f"  Hard-clipped {n_clipped} samples ({n_clipped/total*100:.2f}%) at peak {peak:.3f}")
    else:
        print(f"  Peak: {peak:.3f} — clean")

    sf.write(output_path, final.T, sr)

    elapsed = time_mod.time() - t0
    dur = final.shape[1] / sr
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.0f}s")
    print(f"  {output_path} ({dur/60:.1f}min)")
    print(f"  A: {a_before.shape[1]/sr:.1f}s | Transition: {n/sr:.1f}s ({loop_bars}x{n_repeats}={loop_bars*n_repeats} bars)")
    print(f"  B: {b_after.shape[1]/sr:.1f}s | Peak: {np.max(np.abs(final)):.4f}")
    print(f"{'=' * 60}")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI DJ v3 — CDJ Stem Technique")
    parser.add_argument("track_a", nargs="?", default="track1.mp3")
    parser.add_argument("track_b", nargs="?", default="track2.mp3")
    parser.add_argument("-o", "--output", default="ai_dj_v3_mix.wav")
    args = parser.parse_args()
    mix_tracks(args.track_a, args.track_b, args.output)
