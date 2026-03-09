import os
import json
import librosa
import numpy as np
# Patch numpy for madmom compatibility (np.int removed in NumPy 2.0+)
np.int = int
np.float = float
np.complex = complex
import soundfile as sf
import time
import hashlib
import subprocess
import tempfile
import shutil
import sys
import torch
import madmom
from scipy import signal as scipy_signal
from typing import TypedDict
from langgraph.graph import StateGraph, END
from google import genai
from dotenv import load_dotenv

# --- Studio Configuration ---
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
STRATEGIST_MODEL = "gemini-2.5-flash"          # Pass 1: audio analysis (11 RPD left)
TECHNICAL_MODEL = "gemini-3.1-flash-lite-preview"  # Pass 2: text-only JSON
CACHE_FILE = "mix_cache.json"
STEM_CACHE_DIR = "stem_cache"


def gemini_generate(model, contents, max_retries=3):
    """Wrapper around Gemini API with automatic retry on rate limits."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                # Extract retry delay if present
                import re
                match = re.search(r'retry(?:Delay)?["\s:in]+(\d+)', err_str, re.IGNORECASE)
                wait = int(match.group(1)) + 5 if match else 30
                print(f"    [!] Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini API failed after {max_retries} retries")

class DJState(TypedDict):
    track_a_path: str
    track_b_path: str
    bpm_a: float
    bpm_b: float
    key_a: str
    key_b: str
    mix_plan: dict
    output_path: str
    force_refresh: bool

# --- Utils ---
def get_mix_hash(path_a, path_b):
    hash_md5 = hashlib.md5()
    for p in [path_a, path_b]:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_file_hash(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def apply_envelope(audio, curve):
    return audio * curve[np.newaxis, :]

def snap_to_beat(sample, sr, bpm):
    """Snap a sample position to the nearest beat boundary."""
    beat_samples = int(60.0 / bpm * sr)
    return round(sample / beat_samples) * beat_samples

def snap_to_bar(sample, sr, bpm, beats_per_bar=4):
    """Snap a sample position to the nearest bar boundary."""
    bar_samples = int(beats_per_bar * 60.0 / bpm * sr)
    return round(sample / bar_samples) * bar_samples

def snap_to_kick(audio_array, sr, target_ms, window_ms=100):
    target_sample = int((target_ms / 1000.0) * sr)
    window_samples = int((window_ms / 1000.0) * sr)
    start = max(0, target_sample - window_samples)
    end = min(audio_array.shape[1], target_sample + window_samples)
    segment = librosa.to_mono(audio_array[:, start:end])
    onsets = librosa.onset.onset_detect(y=segment, sr=sr, units='samples', backtrack=True)
    if len(onsets) > 0:
        center_offset = window_samples if start > 0 else target_sample
        closest_onset = min(onsets, key=lambda x: abs(x - center_offset))
        return start + closest_onset
    return start + np.argmax(np.abs(segment))

# ==================================================================
#  DJ EFFECTS ENGINE v3 — resonant filters, send/return reverb,
#  beat-synced modulation, natural tails
# ==================================================================

def apply_resonant_highpass_sweep(audio, sr, bpm, freq_start, freq_end, n_samples):
    """Resonant HP sweep with exponential curve + beat-synced modulation."""
    if freq_end <= 20 and freq_start <= 20:
        return audio

    n_chunks = 64
    chunk_size = n_samples // n_chunks
    result = np.zeros_like(audio)
    beat_samples = int(60.0 / bpm * sr)

    for i in range(n_chunks):
        s0 = i * chunk_size
        s1 = min(s0 + chunk_size, n_samples)
        if s1 <= s0: continue

        t = i / max(n_chunks - 1, 1)
        t_exp = t ** 2.5
        base_freq = freq_start + (freq_end - freq_start) * t_exp

        # Beat-synced wobble
        chunk_center = (s0 + s1) // 2
        beat_phase = (chunk_center % beat_samples) / beat_samples
        modulation = 1.0 + 0.12 * np.sin(2 * np.pi * beat_phase)
        freq = max(25, min(base_freq * modulation, sr * 0.45))

        chunk = audio[:, s0:s1]
        if freq > 30:
            sos_hp = scipy_signal.butter(3, freq, btype='high', fs=sr, output='sos')
            # Resonant peak at cutoff
            bw = freq * 0.15
            low_bp = max(25, freq - bw)
            high_bp = min(sr * 0.45, freq + bw)
            for ch in range(chunk.shape[0]):
                filtered = scipy_signal.sosfilt(sos_hp, chunk[ch])
                if high_bp > low_bp:
                    sos_bp = scipy_signal.butter(2, [low_bp, high_bp], btype='band', fs=sr, output='sos')
                    filtered += scipy_signal.sosfilt(sos_bp, chunk[ch]) * 0.25 * t_exp
                result[ch, s0:s1] = filtered
        else:
            result[:, s0:s1] = chunk
    return result


def apply_resonant_lowpass_sweep(audio, sr, bpm, freq_start, freq_end, n_samples):
    """Resonant LP sweep — muffled opening up."""
    if freq_end >= sr * 0.45 and freq_start >= sr * 0.45:
        return audio

    n_chunks = 64
    chunk_size = n_samples // n_chunks
    result = np.zeros_like(audio)

    for i in range(n_chunks):
        s0 = i * chunk_size
        s1 = min(s0 + chunk_size, n_samples)
        if s1 <= s0: continue
        t = (i / max(n_chunks - 1, 1)) ** 1.8
        freq = max(100, min(freq_start + (freq_end - freq_start) * t, sr * 0.45))

        chunk = audio[:, s0:s1]
        if freq < sr * 0.4:
            sos = scipy_signal.butter(3, freq, btype='low', fs=sr, output='sos')
            for ch in range(chunk.shape[0]):
                result[ch, s0:s1] = scipy_signal.sosfilt(sos, chunk[ch])
        else:
            result[:, s0:s1] = chunk
    return result


def generate_reverb_send(audio, sr, decay_time=2.0):
    """SEND/RETURN reverb — returns WET signal only. Dry stays untouched."""
    n_ch, n_samples = audio.shape
    tap_ms = [11, 23, 37, 53, 71, 97, 131, 173, 229, 307, 401]
    tap_gains = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
    decay_factor = min(decay_time / 2.0, 1.0)

    wet = np.zeros_like(audio)
    for delay_ms, gain in zip(tap_ms, tap_gains):
        delay_samples = int(delay_ms / 1000.0 * sr)
        if delay_samples >= n_samples: continue
        wet[:, delay_samples:] += audio[:, :n_samples - delay_samples] * gain * decay_factor

    # Allpass diffusion
    for ap_ms in [5, 12, 27]:
        ap_samples = int(ap_ms / 1000.0 * sr)
        if ap_samples >= n_samples: continue
        g = 0.6
        diffused = np.zeros_like(wet)
        diffused[:, ap_samples:] = wet[:, :n_samples - ap_samples] * g
        diffused[:, :n_samples - ap_samples] += wet[:, ap_samples:] * (-g)
        diffused += wet * (1 - g * g)
        wet = diffused

    # Dark reverb
    for ch in range(n_ch):
        sos = scipy_signal.butter(2, 3000, btype='low', fs=sr, output='sos')
        wet[ch] = scipy_signal.sosfilt(sos, wet[ch])
    return wet


def generate_reverb_tail(stem_audio_dict, sr, tail_seconds=4.0, feed_seconds=2.0):
    """Generate a natural reverb tail from INDIVIDUAL Track A stems (pre-fade).

    Takes the last `feed_seconds` of each stem, applies heavy reverb,
    and returns a tail that decays naturally. This is generated from
    RAW stem audio, not the faded mix, so it's actually audible.
    """
    tail_samples = int(tail_seconds * sr)
    n_ch = 2

    combined_tail = np.zeros((n_ch, tail_samples))

    # Weight each stem differently for the tail
    stem_weights = {'vocals': 0.5, 'other': 0.6, 'drums': 0.15, 'bass': 0.1}

    for stem_name, stem_audio in stem_audio_dict.items():
        feed_samples = min(int(feed_seconds * sr), stem_audio.shape[1])
        feed = stem_audio[:, -feed_samples:]

        # Extend with silence for tail
        extended = np.zeros((n_ch, feed_samples + tail_samples))
        extended[:, :feed_samples] = feed

        # Heavy reverb
        wet = generate_reverb_send(extended, sr, decay_time=tail_seconds)

        # Extract just the tail portion
        tail = wet[:, feed_samples:]

        # Exponential decay
        decay_env = np.exp(-2.5 * np.linspace(0, 1, tail_samples))
        tail *= decay_env[np.newaxis, :]

        combined_tail += tail * stem_weights.get(stem_name, 0.3)

    return combined_tail


def apply_delay(audio, sr, bpm, feedback=0.4, wet=0.3, pattern="quarter"):
    """Tempo-synced delay with darkening feedback."""
    if wet < 0.01:
        return audio
    beat_sec = 60.0 / bpm
    delay_map = {
        'quarter': beat_sec, 'eighth': beat_sec / 2,
        'dotted_eighth': beat_sec * 0.75, 'triplet': beat_sec / 3,
    }
    delay_samples = int(delay_map.get(pattern, beat_sec / 2) * sr)
    n_ch, n_samples = audio.shape

    sos_dark = scipy_signal.butter(1, 2500, btype='low', fs=sr, output='sos')
    echo_bus = np.zeros_like(audio)
    current_echo = audio.copy()

    for tap in range(6):
        offset = delay_samples * (tap + 1)
        if offset >= n_samples: break
        gain = feedback ** (tap + 1)
        for ch in range(n_ch):
            current_echo[ch] = scipy_signal.sosfilt(sos_dark, current_echo[ch])
        echo_bus[:, offset:] += current_echo[:, :n_samples - offset] * gain

    return audio + echo_bus * wet


def apply_effects_to_stem(audio, sr, bpm, effects, n_samples):
    """Apply effects with send/return architecture."""
    dry = audio.copy()
    wet_bus = np.zeros_like(audio)

    for fx in effects:
        fx_type = fx.get('type', '')
        if fx_type == 'highpass_sweep':
            dry = apply_resonant_highpass_sweep(dry, sr, bpm,
                fx.get('freq_start', 20), fx.get('freq_end', 2000), n_samples)
        elif fx_type == 'lowpass_sweep':
            dry = apply_resonant_lowpass_sweep(dry, sr, bpm,
                fx.get('freq_start', 20000), fx.get('freq_end', 500), n_samples)
        elif fx_type == 'delay':
            dry = apply_delay(dry, sr, bpm,
                feedback=fx.get('feedback', 0.4), wet=fx.get('wet', 0.3),
                pattern=fx.get('pattern', 'dotted_eighth'))
        elif fx_type in ('reverb', 'reverb_wash'):
            wet_start = fx.get('wet', fx.get('wet_start', 0.2))
            wet_end = fx.get('wet', fx.get('wet_end', 0.6))
            decay = fx.get('decay_time', 2.0)
            reverb_send = generate_reverb_send(dry, sr, decay_time=decay)
            wet_env = np.linspace(wet_start, wet_end, n_samples)
            wet_bus += reverb_send * wet_env[np.newaxis, :]

    return dry + wet_bus


# --- DJtransGAN Curve Extraction ---
def get_djtransgan_curves(track_a_path, track_b_path, a_cue_ms, b_cue_ms):
    print("[*] Running DJtransGAN Inference...")
    a_cue_s = a_cue_ms / 1000.0
    b_cue_s = b_cue_ms / 1000.0
    
    # We must run this from the DJtransGAN folder so python can find the `djtransgan` module
    script_cwd = os.path.join(os.getcwd(), "DJtransGAN")

    cmd = [
        sys.executable, "script/inference.py",
        "--prev_track", os.path.abspath(track_a_path),
        "--next_track", os.path.abspath(track_b_path),
        "--prev_cue", str(a_cue_s),
        "--next_cue", str(b_cue_s),
        "--download", "0",
        "--out_dir", "results/inference"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_cwd)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"DJtransGAN failed: {result.stderr}")
        
    saved_id = f"{os.path.splitext(os.path.basename(track_a_path))[0]}_{os.path.splitext(os.path.basename(track_b_path))[0]}"
    curves_path = os.path.join(script_cwd, "results", "inference", f"{saved_id}_curves.pt")
    
    curves = torch.load(curves_path, map_location='cpu')
    return curves


# --- Loop Engine ---
def loop_audio_section(audio, sr, bpm, loop_start_ms, loop_end_ms, n_repeats):
    """Loop a section of audio (like a CDJ loop button).

    Returns new audio with the specified section repeated n_repeats times,
    followed by the rest of the track after loop_end.
    """
    loop_start = snap_to_bar(int(loop_start_ms / 1000.0 * sr), sr, bpm)
    loop_end = snap_to_bar(int(loop_end_ms / 1000.0 * sr), sr, bpm)

    if loop_end <= loop_start or loop_start < 0:
        return audio

    before = audio[:, :loop_start]
    loop_section = audio[:, loop_start:loop_end]
    after = audio[:, loop_end:]

    # Repeat the loop section
    looped = np.tile(loop_section, (1, n_repeats))

    # Small crossfade at loop boundaries to avoid clicks (50ms)
    xfade = min(int(0.05 * sr), loop_section.shape[1] // 4)
    if xfade > 10:
        fade_out = np.linspace(1, 0, xfade)
        fade_in = np.linspace(0, 1, xfade)
        for rep in range(1, n_repeats):
            pos = rep * loop_section.shape[1]
            looped[:, pos - xfade:pos] *= fade_out[np.newaxis, :]
            looped[:, pos:pos + xfade] *= fade_in[np.newaxis, :]

    return np.concatenate([before, looped, after], axis=1)


def loop_stems(stems, sr, bpm, loop_start_ms, loop_end_ms, n_repeats):
    """Apply looping to all stems of a track."""
    result = {}
    for name, audio in stems.items():
        result[name] = loop_audio_section(audio, sr, bpm, loop_start_ms, loop_end_ms, n_repeats)
    return result


# --- Stem Separation with Caching ---
def separate_stems(track_path, target_sr=None):
    file_hash = get_file_hash(track_path)
    cache_dir = os.path.join(STEM_CACHE_DIR, file_hash)
    stem_names = ['drums', 'bass', 'vocals', 'other']

    if os.path.isdir(cache_dir) and all(
        os.path.exists(os.path.join(cache_dir, f"{s}.wav")) for s in stem_names
    ):
        print(f"[*] Stem cache hit for {os.path.basename(track_path)}")
        stems = {}
        for name in stem_names:
            audio, file_sr = sf.read(os.path.join(cache_dir, f"{name}.wav"), always_2d=True)
            audio = audio.T
            if target_sr and file_sr != target_sr:
                channels = [librosa.resample(audio[ch], orig_sr=file_sr, target_sr=target_sr)
                            for ch in range(audio.shape[0])]
                audio = np.stack(channels, axis=0)
            stems[name] = audio
        print(f"[*] Stems loaded: {stems['drums'].shape[1]} samples")
        return stems

    tmp_dir = tempfile.mkdtemp(prefix="dj_stems_")
    track_name = os.path.splitext(os.path.basename(track_path))[0]
    print(f"[*] Separating stems: {os.path.basename(track_path)}...")
    cmd = ["python", "-m", "demucs", "-n", "htdemucs", "--out", tmp_dir, os.path.abspath(track_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed (exit {result.returncode}):\nstderr: {result.stderr}")

    stem_dir = os.path.join(tmp_dir, "htdemucs", track_name)
    stems = {}
    os.makedirs(cache_dir, exist_ok=True)
    for name in stem_names:
        shutil.copy2(os.path.join(stem_dir, f"{name}.wav"), os.path.join(cache_dir, f"{name}.wav"))
        audio, file_sr = sf.read(os.path.join(stem_dir, f"{name}.wav"), always_2d=True)
        audio = audio.T
        if target_sr and file_sr != target_sr:
            channels = [librosa.resample(audio[ch], orig_sr=file_sr, target_sr=target_sr)
                        for ch in range(audio.shape[0])]
            audio = np.stack(channels, axis=0)
        stems[name] = audio

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[*] Stems cached: {stems['drums'].shape[1]} samples")
    return stems

def stretch_audio_with_timemap(audio, sr, timemap_lines, overall_ratio, pitch_semitones=0):
    tmp_dir = tempfile.mkdtemp(prefix="dj_rb_")
    tmp_in = os.path.join(tmp_dir, "in.wav")
    tmp_out = os.path.join(tmp_dir, "out.wav")
    tmp_map = os.path.join(tmp_dir, "map.txt")
    try:
        sf.write(tmp_in, audio.T, sr)
        with open(tmp_map, 'w') as f: f.write("\n".join(timemap_lines) + "\n")
        rb_cmd = ["rubberband", "-t", str(overall_ratio), "--timemap", tmp_map]
        if pitch_semitones != 0: rb_cmd.extend(["--pitch", str(pitch_semitones)])
        rb_cmd.extend([tmp_in, tmp_out])
        result = subprocess.run(rb_cmd, capture_output=True, text=True)
        if result.returncode != 0: raise RuntimeError(f"rubberband failed: {result.stderr}")
        stretched, _ = sf.read(tmp_out, always_2d=True)
        return stretched.T
    finally:
        for p in [tmp_in, tmp_out, tmp_map]:
            if os.path.exists(p): os.remove(p)
        if os.path.exists(tmp_dir): os.rmdir(tmp_dir)

# --- Structural Pre-Analysis ---
def analyze_track_structure(audio_path, bpm):
    """Analyze track structure using librosa. Returns energy curve,
    breakdown positions, and vocal activity map."""
    y, sr = librosa.load(audio_path, sr=22050)
    duration_ms = len(y) / sr * 1000

    # RMS energy in 1-second windows
    hop = sr  # 1 second
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)

    # Find energy drops (potential breakdowns) — where energy drops >30% from local average
    breakdowns = []
    window = 8  # 8-second moving average
    for i in range(window, len(rms_norm) - window):
        local_avg = np.mean(rms_norm[max(0, i-window):i])
        if local_avg > 0.4 and rms_norm[i] < local_avg * 0.65:
            ms = int(i * 1000)
            # Only keep breakdowns that are 40-85% through the track
            pct = ms / duration_ms
            if 0.4 < pct < 0.85:
                breakdowns.append({'ms': ms, 'energy_drop': float(local_avg - rms_norm[i]),
                                   'position_pct': round(pct * 100)})

    # Deduplicate (keep strongest breakdown per 10-second window)
    filtered = []
    for bd in sorted(breakdowns, key=lambda x: -x['energy_drop']):
        if not any(abs(bd['ms'] - f['ms']) < 10000 for f in filtered):
            filtered.append(bd)
    breakdowns = sorted(filtered, key=lambda x: x['ms'])

    # Detect vocal activity using spectral flatness (vocal sections are less flat)
    spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
    # Lower spectral flatness = more tonal/vocal content
    vocal_activity = []
    for i in range(0, len(spec_flat), 4):  # 4-second chunks
        chunk = spec_flat[i:i+4]
        is_vocal = float(np.mean(chunk)) < 0.15  # Threshold for vocal detection
        vocal_activity.append({'time_s': i, 'has_vocals': is_vocal})

    return {
        'duration_ms': int(duration_ms),
        'breakdowns': breakdowns[:5],  # Top 5
        'energy_summary': [round(float(rms_norm[int(p * len(rms_norm))]), 2)
                          for p in [0.1, 0.25, 0.5, 0.75, 0.9]],
    }


def validate_and_fix_plan(plan, track_a_duration_ms, track_b_duration_ms, bpm_a):
    """Validate and auto-correct a mix plan."""
    issues = []
    bar_ms = 4 * 60000 / bpm_a  # Duration of 1 bar in ms

    # 1. Track A breakdown must be 40-85% through
    bd = plan['track_a_breakdown_start_ms']
    pct = bd / track_a_duration_ms
    if pct > 0.85:
        new_bd = int(track_a_duration_ms * 0.65)
        issues.append(f'Breakdown too late ({pct:.0%}) -> moved to 65%')
        plan['track_a_breakdown_start_ms'] = new_bd
    elif pct < 0.35:
        new_bd = int(track_a_duration_ms * 0.55)
        issues.append(f'Breakdown too early ({pct:.0%}) -> moved to 55%')
        plan['track_a_breakdown_start_ms'] = new_bd

    # 2. Transition must be 16-32 bars
    trans_ms = plan['track_b_drop_ms'] - plan['track_b_intro_start_ms']
    min_trans = 16 * bar_ms
    max_trans = 40 * bar_ms
    if trans_ms < min_trans:
        plan['track_b_drop_ms'] = int(plan['track_b_intro_start_ms'] + 24 * bar_ms)
        issues.append(f'Transition too short ({trans_ms/1000:.1f}s) -> set to 24 bars')
    elif trans_ms > max_trans:
        plan['track_b_drop_ms'] = int(plan['track_b_intro_start_ms'] + 24 * bar_ms)
        issues.append(f'Transition too long ({trans_ms/1000:.1f}s) -> set to 24 bars')

    # 3. Track B drop can't exceed track length
    if plan['track_b_drop_ms'] > track_b_duration_ms * 0.5:
        plan['track_b_drop_ms'] = int(min(track_b_duration_ms * 0.3, plan['track_b_intro_start_ms'] + 24 * bar_ms))
        issues.append('Track B drop too late -> capped')

    # 4. Ensure intro < drop (critical: negative transition = crash)
    if plan['track_b_intro_start_ms'] >= plan['track_b_drop_ms']:
        # Reset intro to 0 and set drop to 24 bars
        plan['track_b_intro_start_ms'] = 0
        plan['track_b_drop_ms'] = int(24 * bar_ms)
        issues.append(f'Intro >= drop! Reset intro=0, drop={plan["track_b_drop_ms"]}ms')

    # 5. Final sanity: transition must be positive and reasonable
    final_trans = plan['track_b_drop_ms'] - plan['track_b_intro_start_ms']
    if final_trans < 10 * bar_ms:
        plan['track_b_drop_ms'] = int(plan['track_b_intro_start_ms'] + 24 * bar_ms)
        issues.append(f'Final transition too short -> forced 24 bars')

    for issue in issues:
        print(f'    [FIX] {issue}')

    return plan


def normalize_stem_effects(plan):
    """Normalize stem_effects: ensure each effect is a dict, not a string."""
    for fx_spec in plan.get('stem_effects', []):
        if not isinstance(fx_spec, dict):
            continue
        normalized = []
        for e in fx_spec.get('effects', []):
            if isinstance(e, str):
                # Convert string like 'highpass_sweep' to dict
                normalized.append({'type': e})
            elif isinstance(e, dict):
                normalized.append(e)
        fx_spec['effects'] = normalized
    # Also normalize stem_events
    for evt in plan.get('stem_events', []):
        if isinstance(evt, dict):
            evt['time_pct'] = float(evt.get('time_pct', 0))
            evt['gain'] = float(evt.get('gain', 0))
    return plan


# --- BPM & Key Detection ---
def detect_bpm_madmom(audio_path):
    proc = madmom.features.beats.RNNBeatProcessor()(audio_path)
    beats = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(proc)
    if len(beats) < 2: raise ValueError(f"Not enough beats in {audio_path}")
    bpm = 60.0 / np.median(np.diff(beats))
    if bpm < 80: bpm *= 2
    elif bpm > 200: bpm /= 2
    return round(bpm, 1)

def analyzer_node(state: DJState):
    print(f"[*] Node: Analyzer | madmom RNN beat tracking + key detection...")
    state['bpm_a'] = detect_bpm_madmom(state['track_a_path'])
    state['bpm_b'] = detect_bpm_madmom(state['track_b_path'])
    y_a, sr_a = librosa.load(state['track_a_path'])
    y_b, sr_b = librosa.load(state['track_b_path'])
    chroma_a = librosa.feature.chroma_stft(y=y_a, sr=sr_a)
    chroma_b = librosa.feature.chroma_stft(y=y_b, sr=sr_b)
    key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    state['key_a'] = key_map[np.argmax(np.mean(chroma_a, axis=1))]
    state['key_b'] = key_map[np.argmax(np.mean(chroma_b, axis=1))]
    bpm_diff = abs(state['bpm_a'] - state['bpm_b']) / max(state['bpm_a'], state['bpm_b']) * 100
    print(f"[*] Track A: {state['key_a']} @ {state['bpm_a']:.1f} BPM")
    print(f"[*] Track B: {state['key_b']} @ {state['bpm_b']:.1f} BPM")
    print(f"[*] BPM difference: {bpm_diff:.1f}%{' [!] Large gap' if bpm_diff > 8 else ' [OK]'}")
    return state

def strategist_node(state: DJState):
    mix_hash = get_mix_hash(state['track_a_path'], state['track_b_path'])

    if not state.get('force_refresh', False) and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                cache = json.load(f)
                if mix_hash in cache:
                    print(f"[*] Node: Strategist | Cache hit.")
                    state['mix_plan'] = cache[mix_hash]
                    return state
            except: pass

    print(f"[*] Node: Strategist | Pre-analyzing track structure...")
    struct_a = analyze_track_structure(state['track_a_path'], state['bpm_a'])
    struct_b = analyze_track_structure(state['track_b_path'], state['bpm_b'])
    bd_strs = [f"{b['ms']/1000:.0f}s ({b['position_pct']}%)" for b in struct_a['breakdowns']]
    print(f"    Track A: {struct_a['duration_ms']/1000:.0f}s, breakdowns: {bd_strs}")
    print(f"    Track B: {struct_b['duration_ms']/1000:.0f}s")

    # Upload files (shared between both passes)
    file_a = client.files.upload(file=state['track_a_path'])
    file_b = client.files.upload(file=state['track_b_path'])
    while file_a.state.name == "PROCESSING" or file_b.state.name == "PROCESSING":
        time.sleep(2)
        file_a = client.files.get(name=file_a.name)
        file_b = client.files.get(name=file_b.name)

    # --- PASS 1: Creative Director (free-form listening) ---
    print(f"[*] Pass 1: Creative Director — listening to both tracks...")
    bd_list = '\n'.join(f"  - {b['ms']}ms ({b['position_pct']}%)" for b in struct_a['breakdowns'])
    creative_prompt = f"""You are a world-class DJ about to mix Track A into Track B live at a packed club.

Track A: key={state['key_a']}, BPM={state['bpm_a']:.1f}, duration={struct_a['duration_ms']/1000:.0f}s
Track A energy drops detected at: {bd_list or 'none found'}
Track B: key={state['key_b']}, BPM={state['bpm_b']:.1f}, duration={struct_b['duration_ms']/1000:.0f}s

Listen to BOTH tracks carefully. Then answer these questions:

1. STRUCTURE: Describe the structure of each track — where are the builds, breakdowns, drops, vocal sections, instrumental sections? Use specific timestamps.

2. VOCAL MAP: When exactly do vocals appear in Track A? When in Track B? Are there instrumental gaps we can exploit?

3. CREATIVE VISION: Describe in 2-3 sentences HOW you would transition. What's the story arc? What's the emotional journey? Be specific to THESE tracks — don't be generic.

4. MIX POINT: At what SPECIFIC moment in Track A would you start mixing out? Why that moment? (Should be during a breakdown or energy dip, not during a vocal hook.)

5. ENERGY: How would you manage energy so the transition feels like one continuous journey, not two songs fighting?

Be specific. Use timestamps. This is a creative brief, not a technical spec."""

    creative_response = gemini_generate(
        model=STRATEGIST_MODEL, contents=[creative_prompt, file_a, file_b]
    )
    creative_vision = creative_response.text.strip()
    print(f"[*] Creative vision received ({len(creative_vision)} chars)")
    # Print a summary (first 300 chars)
    print(f"    \"{creative_vision[:300]}...\"")

    # --- PASS 2: Technical Engineer (structured output) ---
    print(f"[*] Pass 2: Technical Engineer — converting vision to mix plan...")
    technical_prompt = f"""You are a mix engineer. A creative director has analyzed two tracks and described their vision.
Convert their vision into a precise technical mix plan.

CREATIVE VISION:
{creative_vision}

TRACK DATA:
Track A: key={state['key_a']}, BPM={state['bpm_a']:.1f}, duration={struct_a['duration_ms']}ms
Track B: key={state['key_b']}, BPM={state['bpm_b']:.1f}, duration={struct_b['duration_ms']}ms

HARD RULES (override the creative vision if it violates these):
- Track A breakdown: 40-85% through Track A. Use the creative director's suggested mix point.
- Transition: 16-24 bars (30-48 seconds at this BPM). NOT longer.
- ZERO VOCAL OVERLAP: Track A vocals must be gain 0.0 BEFORE Track B vocals go above 0.0. Leave a gap.
- BASS: Never overlap. Track A bass cuts (sudden), silence gap, Track B bass drops at 100%.
- DRUMS: Track A drums hold then cut suddenly (not slow fade). Track B drums can layer hi-hats early.
- High frequencies: When Track B drums start coming in, apply highpass_sweep to Track A drums to avoid clash.

AVAILABLE FX (each applied to a specific stem over a % range of the transition):
- "highpass_sweep": freq_start/freq_end Hz — thins out a stem
- "lowpass_sweep": freq_start/freq_end Hz — muffled sound opening up  
- "reverb_wash": wet_start/wet_end (keep under 0.4) — dissolves into space
- "delay": wet (under 0.3)/feedback/pattern — rhythmic echo trail

Return ONLY this JSON:
{{
  "track_a_breakdown_start_ms": <ms>,
  "track_b_intro_start_ms": <ms>,
  "track_b_drop_ms": <ms>,
  "pitch_shift_semitones": <int>,
  "transition_style": "<2-3 words>",
  "track_b_loop": null,
  "creative_notes": "<1 sentence summary of creative vision>",
  "stem_events": [
    ... gain automation events with time_pct (0-100), track (a/b), stem, gain (0-1), curve ...
  ],
  "stem_effects": [
    ... FX chains with track, stem, phase_start_pct, phase_end_pct, effects list ...
  ]
}}

Curves: "linear", "exponential" (DJ-style sweep), "sudden" (hard cut with 50ms anti-click).
Stems: "drums", "bass", "vocals", "other".

REMEMBER: Vocal overlap = instant fail. Bass overlap = instant fail. Be decisive, not gradual."""

    technical_response = gemini_generate(
        model=TECHNICAL_MODEL, contents=[technical_prompt]
    )

    client.files.delete(name=file_a.name)
    client.files.delete(name=file_b.name)

    text = technical_response.text.strip()
    if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text: text = text.split("```")[1].split("```")[0].strip()

    plan = json.loads(text)

    # Normalize effects format
    plan = normalize_stem_effects(plan)

    # Validate and auto-correct
    plan = validate_and_fix_plan(
        plan, struct_a['duration_ms'], struct_b['duration_ms'], state['bpm_a']
    )

    state['mix_plan'] = plan
    print(f"[*] Mix plan: style='{plan.get('transition_style', 'N/A')}'")
    if plan.get('creative_notes'):
        print(f"    Vision: {plan['creative_notes']}")
    print(f"    Track A breakdown: {plan['track_a_breakdown_start_ms']}ms ({plan['track_a_breakdown_start_ms']/struct_a['duration_ms']*100:.0f}%)")
    print(f"    Track B intro: {plan['track_b_intro_start_ms']}ms -> drop: {plan['track_b_drop_ms']}ms")
    trans_ms = plan['track_b_drop_ms'] - plan['track_b_intro_start_ms']
    print(f"    Transition: {trans_ms/1000:.1f}s ({trans_ms/1000/60*state['bpm_a']/4:.0f} bars)")
    if plan.get('track_b_loop'):
        lp = plan['track_b_loop']
        print(f"    Loop: Track B {lp['loop_start_ms']}-{lp['loop_end_ms']}ms x{lp['n_repeats']}")
    print(f"    Stem events: {len(plan.get('stem_events', []))}, FX: {len(plan.get('stem_effects', []))}")
    for fx in plan.get('stem_effects', []):
        try:
            if isinstance(fx, dict):
                fx_types = [e['type'] if isinstance(e, dict) else str(e) for e in fx.get('effects', [])]
                print(f"      {fx['track'].upper()}.{fx['stem']}: {', '.join(fx_types)} "
                      f"({fx.get('phase_start_pct', '?')}%-{fx.get('phase_end_pct', '?')}%)")
        except (KeyError, TypeError):
            print(f"      [?] Malformed FX entry: {fx}")
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try: cache = json.load(f)
            except: pass
    cache[mix_hash] = plan
    with open(CACHE_FILE, 'w') as f: json.dump(cache, f, indent=2)
    return state


# --- Envelope Builder ---
def build_stem_envelope(stem_events, n_samples, track, stem, bpm=None, sr=None):
    events = [e for e in stem_events if e['track'] == track and e['stem'] == stem]
    events.sort(key=lambda e: e['time_pct'])

    if not events:
        return np.ones(n_samples) if track == 'a' else np.zeros(n_samples)

    if events[0]['time_pct'] > 0:
        events.insert(0, {'time_pct': 0, 'gain': 1.0 if track == 'a' else 0.0, 'curve': 'linear'})
    if events[-1]['time_pct'] < 100:
        events.append({'time_pct': 100, 'gain': 0.0 if track == 'a' else 1.0, 'curve': 'linear'})

    envelope = np.zeros(n_samples)

    for i in range(len(events) - 1):
        s0 = max(0, min(int(events[i]['time_pct'] / 100.0 * n_samples), n_samples))
        s1 = max(0, min(int(events[i + 1]['time_pct'] / 100.0 * n_samples), n_samples))
        if s1 <= s0: continue

        seg_len = s1 - s0
        g0, g1 = float(events[i]['gain']), float(events[i + 1]['gain'])
        curve = events[i + 1].get('curve', 'linear')

        if curve == 'sudden':
            # Hold then snap — but make the snap happen over ~50ms to avoid click
            click_guard = min(int(0.05 * sr) if sr else 100, seg_len // 2)
            seg = np.full(seg_len, g0)
            seg[-click_guard:] = np.linspace(g0, g1, click_guard)
        elif curve == 'exponential':
            t = np.linspace(0, 1, seg_len)
            if g1 > g0: seg = g0 + (g1 - g0) * (t ** 2.5)
            else: seg = g0 + (g1 - g0) * (1 - (1 - t) ** 2.5)
        else:
            seg = np.linspace(g0, g1, seg_len)

        envelope[s0:s1] = seg

    last = int(events[-1]['time_pct'] / 100.0 * n_samples)
    if last < n_samples:
        envelope[last:] = events[-1]['gain']

    return np.clip(envelope, 0.0, 1.0)


def enforce_hard_vocal_separation(envelopes, n):
    """HARD rule: at any sample, only ONE track's vocals can be non-zero.

    When both are non-zero, the outgoing (Track A) takes priority in the first half,
    incoming (Track B) takes priority in the second half. The loser goes to 0.
    """
    a_vox = envelopes['a_vocals'].copy()
    b_vox = envelopes['b_vocals'].copy()

    # Find the crossover point (where A drops below B)
    crossover = n // 2  # Default to midpoint
    for i in range(n):
        if a_vox[i] < b_vox[i]:
            crossover = i
            break

    # In the first half: A has priority, B must be 0 wherever A > 0.05
    # In the second half: B has priority, A must be 0 wherever B > 0.05
    for i in range(n):
        if i < crossover:
            if a_vox[i] > 0.05:
                b_vox[i] = 0.0
        else:
            if b_vox[i] > 0.05:
                a_vox[i] = 0.0

    # Smooth the cuts to avoid clicks (20ms ramp)
    smooth_samples = min(int(0.02 * 48000), n // 10)
    for arr in [a_vox, b_vox]:
        for i in range(1, n):
            if arr[i] == 0.0 and arr[i-1] > 0.05:
                # Find where the cut starts and apply tiny ramp
                ramp_end = min(i + smooth_samples, n)
                arr[i:ramp_end] = np.linspace(arr[i-1], 0.0, ramp_end - i)

    envelopes['a_vocals'] = a_vox
    envelopes['b_vocals'] = b_vox
    return envelopes


def enforce_bass_exclusion(envelopes, n):
    """HARD bass swap: never overlap above 0.3 combined."""
    a_bass = envelopes['a_bass'].copy()
    b_bass = envelopes['b_bass'].copy()

    for i in range(n):
        total = a_bass[i] + b_bass[i]
        if total > 0.3 and a_bass[i] > 0.05 and b_bass[i] > 0.05:
            # Whoever is louder wins, other goes to 0
            if a_bass[i] >= b_bass[i]:
                b_bass[i] = 0.0
            else:
                a_bass[i] = 0.0

    envelopes['a_bass'] = a_bass
    envelopes['b_bass'] = b_bass
    return envelopes


def renderer_node(state: DJState):
    print(f"[*] Node: Renderer | Stem-choreographed mix with FX + natural tails...")
    plan = state['mix_plan']

    # ---------------------------------------------------------------
    # 0. Load audio
    # ---------------------------------------------------------------
    y_a, sr = sf.read(state['track_a_path'], always_2d=True); y_a = y_a.T
    y_b_raw, sr_b = sf.read(state['track_b_path'], always_2d=True); y_b_raw = y_b_raw.T
    print(f"[*] Track A: {y_a.shape[1]/sr:.1f}s @ {sr}Hz | Track B: {y_b_raw.shape[1]/sr_b:.1f}s @ {sr_b}Hz")

    if sr_b != sr:
        print(f"[!] Resampling Track B {sr_b} -> {sr}Hz")
        y_b_raw = np.stack([librosa.resample(y_b_raw[ch], orig_sr=sr_b, target_sr=sr)
                            for ch in range(y_b_raw.shape[0])], axis=0)
    else:
        print(f"[*] Sample rates match: {sr}Hz")

    # ---------------------------------------------------------------
    # 1. Separate stems (cached)
    # ---------------------------------------------------------------
    stems_a = separate_stems(state['track_a_path'], target_sr=sr)
    stems_b = separate_stems(state['track_b_path'], target_sr=sr)

    # ---------------------------------------------------------------
    # 1b. Apply loops if specified
    # ---------------------------------------------------------------
    if plan.get('track_b_loop'):
        lp = plan['track_b_loop']
        print(f"[*] Looping Track B: {lp['loop_start_ms']}-{lp['loop_end_ms']}ms x{lp['n_repeats']}")
        stems_b = loop_stems(stems_b, sr, state['bpm_b'],
                             lp['loop_start_ms'], lp['loop_end_ms'], lp['n_repeats'])
        # Also loop the full audio for reconstruction
        y_b_raw = loop_audio_section(y_b_raw, sr, state['bpm_b'],
                                     lp['loop_start_ms'], lp['loop_end_ms'], lp['n_repeats'])
        print(f"[*] Track B after loop: {y_b_raw.shape[1]/sr:.1f}s")

    # ---------------------------------------------------------------
    # 2. Timemap for tempo drift
    # ---------------------------------------------------------------
    stretch_ratio = state['bpm_b'] / state['bpm_a']
    pitch_semitones = plan.get('pitch_shift_semitones', 0)
    total_src = int(y_b_raw.shape[1])
    b_drop_src = int((plan['track_b_drop_ms'] / 1000.0) * sr)

    drift_beats = 32 * 4
    drift_samples = int(drift_beats * (60.0 / state['bpm_b']) * sr)
    b_drift_end_src = int(min(b_drop_src + drift_samples, total_src))

    R0, R1 = stretch_ratio, 1.0
    drift_len = b_drift_end_src - b_drop_src
    b_drop_tgt = int(b_drop_src * stretch_ratio)

    drift_points = []
    for i in range(129):
        s = drift_len * i / 128
        tgt_off = R0 * s + (R1 - R0) * s * s / (2.0 * drift_len) if drift_len > 0 else s
        drift_points.append((int(b_drop_src + s), int(b_drop_tgt + tgt_off)))

    b_drift_end_tgt = drift_points[-1][1]
    total_tgt = int(b_drift_end_tgt + (total_src - b_drift_end_src))
    timemap_lines = ["0 0"] + [f"{s} {t}" for s, t in drift_points] + [f"{total_src} {total_tgt}"]
    overall_ratio = total_tgt / total_src
    print(f"[*] Stretch: {stretch_ratio:.4f} ({state['bpm_b']:.1f} -> {state['bpm_a']:.1f} BPM)")

    # ---------------------------------------------------------------
    # 3. Stretch Track B stems
    # ---------------------------------------------------------------
    stems_b_stretched = {}
    for name, audio in stems_b.items():
        t0 = time.time()
        stems_b_stretched[name] = stretch_audio_with_timemap(audio, sr, timemap_lines, overall_ratio, pitch_semitones)
        print(f"[*] Stretched B.{name}: {stems_b_stretched[name].shape[1]} ({time.time()-t0:.1f}s)")

    y_b_final = sum(stems_b_stretched.values())

    # ---------------------------------------------------------------
    # 4. Transition boundaries
    # ---------------------------------------------------------------
    a_start = snap_to_kick(y_a, sr, plan['track_a_breakdown_start_ms'])
    b_start = snap_to_kick(y_b_final, sr, plan['track_b_intro_start_ms'] * stretch_ratio)
    b_drop = snap_to_kick(y_b_final, sr, (b_drop_tgt / sr) * 1000.0)

    trans_len = b_drop - b_start
    bars = trans_len / sr / 60 * state['bpm_a'] / 4
    print(f"[*] Transition: {trans_len/sr:.1f}s ({bars:.0f} bars)")
    if trans_len <= 0: raise ValueError(f"Invalid transition: {trans_len}")

    # ---------------------------------------------------------------
    # 5. Slice transition stems
    # ---------------------------------------------------------------
    a_trans, b_trans = {}, {}
    for s in ['drums', 'bass', 'vocals', 'other']:
        a_trans[s] = stems_a[s][:, a_start : a_start + trans_len]
        b_trans[s] = stems_b_stretched[s][:, b_start : b_drop]

    n = min(*(a_trans[s].shape[1] for s in a_trans), *(b_trans[s].shape[1] for s in b_trans))
    for s in ['drums', 'bass', 'vocals', 'other']:
        a_trans[s] = a_trans[s][:, :n]
        b_trans[s] = b_trans[s][:, :n]

    # ---------------------------------------------------------------
    # 6. Apply performance effects (send/return)
    # ---------------------------------------------------------------
    stem_effects = plan.get('stem_effects', [])
    if stem_effects:
        print(f"[*] Applying {len(stem_effects)} FX chains...")
    bpm_fx = state['bpm_a']

    for fx_spec in stem_effects:
        track, stem = fx_spec['track'], fx_spec['stem']
        s0 = max(0, min(int(fx_spec.get('phase_start_pct', 0) / 100.0 * n), n))
        s1 = max(0, min(int(fx_spec.get('phase_end_pct', 100) / 100.0 * n), n))
        if s1 <= s0: continue

        stems_dict = a_trans if track == 'a' else b_trans
        if stem not in stems_dict: continue

        fx_list = fx_spec.get('effects', [])
        # Ensure all effects are dicts
        fx_list = [e if isinstance(e, dict) else {'type': str(e)} for e in fx_list]
        fx_names = [f.get('type', '?') if isinstance(f, dict) else str(f) for f in fx_list]
        print(f"    {track.upper()}.{stem} [{fx_spec.get('phase_start_pct',0)}-{fx_spec.get('phase_end_pct',100)}%]: "
              f"{', '.join(fx_names)}")

        processed = apply_effects_to_stem(stems_dict[stem][:, s0:s1].copy(), sr, bpm_fx, fx_list, s1 - s0)
        stems_dict[stem][:, s0:s1] = processed

    # ---------------------------------------------------------------
    # 7. Build envelopes + enforce HARD constraints or DJtransGAN
    # ---------------------------------------------------------------
    stem_events = plan.get('stem_events', [])
    print(f"[*] Building envelopes ({len(stem_events)} events)...")

    envelopes = {}
    for track in ['a', 'b']:
        for stem in ['drums', 'bass', 'vocals', 'other']:
            envelopes[f"{track}_{stem}"] = build_stem_envelope(
                stem_events, n, track, stem, bpm=state['bpm_a'], sr=sr)

    # Attempt to inject DJtransGAN Fader curves
    try:
        curves = get_djtransgan_curves(state['track_a_path'], state['track_b_path'],
                                       plan['track_a_breakdown_start_ms'], 
                                       plan['track_b_intro_start_ms'] * stretch_ratio)
        
        # DJtransGAN outputs curves sized 5168. We stretch them to `n` (our transition samples).
        prev_fader = curves['prev']['fader'].squeeze().numpy()  # [4, 5168]
        next_fader = curves['next']['fader'].squeeze().numpy()  # [4, 5168]

        def resample_curve(curve, target_len):
            return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(curve)), curve)

        print(f"[*] Substituting DJtransGAN Neural Fader Curves!")
        # Map 4 DJtransGAN bands:
        # 0 = Low (Bass), 1 = Mid (Vocals/Other), 2 = High (Drums), 3 = Master (apply to all slightly?)
        
        # Track A
        envelopes['a_bass']   = resample_curve(prev_fader[0], n)
        envelopes['a_vocals'] = resample_curve(prev_fader[1], n)
        envelopes['a_other']  = resample_curve(prev_fader[1], n)
        envelopes['a_drums']  = resample_curve(prev_fader[2], n)

        # Track B
        envelopes['b_bass']   = resample_curve(next_fader[0], n)
        envelopes['b_vocals'] = resample_curve(next_fader[1], n)
        envelopes['b_other']  = resample_curve(next_fader[1], n)
        envelopes['b_drums']  = resample_curve(next_fader[2], n)

    except Exception as e:
        print(f"[!] DJtransGAN integration failed. Falling back to Gemini plan. Error: {e}")

    # Enforce HARD separation rules
    envelopes = enforce_hard_vocal_separation(envelopes, n)
    envelopes = enforce_bass_exclusion(envelopes, n)

    print(f"[*] Envelopes (0% / 25% / 50% / 75% / 100%):")
    for track in ['a', 'b']:
        for stem in ['drums', 'bass', 'vocals', 'other']:
            env = envelopes[f"{track}_{stem}"]
            vals = [env[min(int(p * n), n - 1)] for p in [0, 0.25, 0.5, 0.75, 1.0]]
            print(f"    {track.upper()}.{stem:6s}: {' / '.join(f'{v:.2f}' for v in vals)}")

    # ---------------------------------------------------------------
    # 8. Mix transition
    # ---------------------------------------------------------------
    transition_block = np.zeros((2, n))
    for stem in ['drums', 'bass', 'vocals', 'other']:
        transition_block += apply_envelope(a_trans[stem], envelopes[f'a_{stem}'])
        transition_block += apply_envelope(b_trans[stem], envelopes[f'b_{stem}'])

    # Soft limiter
    peak = np.max(np.abs(transition_block))
    if peak > 0.92:
        print(f"[*] Transition peak {peak:.3f} — soft limiting")
        transition_block = np.tanh(transition_block / peak * 0.92) * 0.92

    # ---------------------------------------------------------------
    # 9. Generate reverb tail from Track A stems (pre-envelope!)
    # ---------------------------------------------------------------
    print(f"[*] Generating reverb tail from Track A stems (pre-fade)...")
    reverb_tail = generate_reverb_tail(a_trans, sr, tail_seconds=4.0, feed_seconds=2.0)
    print(f"[*] Reverb tail: {reverb_tail.shape[1]/sr:.1f}s")

    # ---------------------------------------------------------------
    # 10. Final assembly with natural tail bleed
    # ---------------------------------------------------------------
    track_b_after = y_b_final[:, b_drop:]
    tail_len = reverb_tail.shape[1]
    blend_len = min(tail_len, track_b_after.shape[1])

    track_b_section = track_b_after.copy()
    # Subtle tail blend (0.25) — enough to avoid abrupt cut, not enough to mud the drop
    track_b_section[:, :blend_len] += reverb_tail[:, :blend_len] * 0.25

    final_mix = np.concatenate((y_a[:, :a_start], transition_block, track_b_section), axis=1)

    # Global limiting
    peak = np.max(np.abs(final_mix))
    if peak > 0.95:
        final_mix = np.tanh(final_mix / peak * 0.93) * 0.93
        print(f"[*] Global peak {peak:.3f} -> limited to 0.93")

    peak = np.max(np.abs(final_mix))
    print(f"[*] Final: {final_mix.shape[1]/sr:.1f}s | Peak: {peak:.4f} {'[OK]' if peak <= 1.0 else '[CLIP]'}")
    print(f"    A: {a_start/sr:.1f}s | Trans: {n/sr:.1f}s | B: {track_b_section.shape[1]/sr:.1f}s | Tail: {tail_len/sr:.1f}s")

    sf.write(state['output_path'], final_mix.T, sr)
    print(f"[SUCCESS] -> {state['output_path']}")
    return state

# --- Graph ---
builder = StateGraph(DJState)
builder.add_node("analyze", analyzer_node)
builder.add_node("strategize", strategist_node)
builder.add_node("render", renderer_node)
builder.set_entry_point("analyze")
builder.add_edge("analyze", "strategize")
builder.add_edge("strategize", "render")
builder.add_edge("render", END)
app = builder.compile()

if __name__ == "__main__":
    input_state = {
        "track_a_path": "track1.mp3",
        "track_b_path": "track2.mp3",
        "output_path": "ai_dj_final_master.wav",
        "force_refresh": True,
    }
    app.invoke(input_state)
