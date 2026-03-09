#!/usr/bin/env python3
"""Deep analysis of both tracks for Claude to DJ with."""

import numpy as np
import librosa
import soundfile as sf
import madmom
import json

np.int = int
np.float = float
np.complex = complex


def analyze_track(path, label):
    print(f"\n{'='*60}")
    print(f"  DEEP ANALYSIS: {label} — {path}")
    print(f"{'='*60}")

    # Load audio
    y, sr = librosa.load(path, sr=22050)
    duration = len(y) / sr
    print(f"\n  Duration: {duration:.1f}s | SR: {sr}")

    # --- BEATS & BARS ---
    print("\n  [Beats]")
    proc = madmom.features.beats.RNNBeatProcessor()(path)
    beats = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)(proc)
    bpm = 60.0 / np.median(np.diff(beats))
    while bpm < 80: bpm *= 2
    while bpm > 200: bpm /= 2
    print(f"  BPM: {bpm:.1f}")
    print(f"  Total beats: {len(beats)}")

    proc_db = madmom.features.downbeats.RNNDownBeatProcessor()(path)
    result_db = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4], fps=100)(proc_db)
    bars = result_db[result_db[:, 1] == 1, 0]
    print(f"  Total bars: {len(bars)}")
    print(f"  Bar duration: {np.median(np.diff(bars)):.2f}s")

    # --- KEY ---
    print("\n  [Key]")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best_corr, best_key, best_mode = -1, 'C', 'major'
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        for profile, mode in [(MAJOR_PROFILE, 'major'), (MINOR_PROFILE, 'minor')]:
            corr = np.corrcoef(rolled, profile)[0, 1]
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, KEY_NAMES[shift], mode
    key_str = best_key if best_mode == 'major' else f'{best_key}m'
    print(f"  Key: {key_str} (confidence: {best_corr:.3f})")

    # --- ENERGY PROFILE (per bar) ---
    print("\n  [Energy per bar]")
    bar_energies = []
    for i in range(len(bars)):
        start = int(bars[i] * sr)
        end = int(bars[i + 1] * sr) if i + 1 < len(bars) else len(y)
        chunk = y[start:end]
        rms = np.sqrt(np.mean(chunk ** 2))
        bar_energies.append(rms)

    # Normalize
    max_e = max(bar_energies) + 1e-8
    bar_energies_norm = [e / max_e for e in bar_energies]

    # Print as visual bars in groups of 4 (phrases)
    for i in range(0, len(bars), 4):
        chunk = bar_energies_norm[i:i+4]
        avg = np.mean(chunk)
        bar_times = f"{bars[i]:.1f}s"
        visual = "#" * int(avg * 40)
        bar_nums = f"bars {i+1}-{min(i+4, len(bars))}"
        print(f"    {bar_nums:>12} ({bar_times:>7}): {visual:<40} {avg:.3f}")

    # --- STRUCTURE DETECTION ---
    print("\n  [Structure — energy sections]")
    # Group bars into 8-bar phrases and classify
    phrases = []
    for i in range(0, len(bars), 8):
        chunk = bar_energies_norm[i:i+8]
        if len(chunk) < 4:
            continue
        avg = np.mean(chunk)
        start_t = bars[i]
        end_t = bars[min(i+8, len(bars)-1)] if i+8 < len(bars) else duration

        if avg > 0.7:
            label_s = "HIGH ENERGY (drop/chorus)"
        elif avg > 0.4:
            label_s = "MID ENERGY (verse/groove)"
        elif avg > 0.15:
            label_s = "LOW ENERGY (breakdown/buildup)"
        else:
            label_s = "SILENCE (intro/outro)"

        phrases.append({
            'bars': f"{i+1}-{i+8}",
            'time': f"{start_t:.1f}-{end_t:.1f}s",
            'energy': avg,
            'type': label_s
        })
        print(f"    Bars {i+1:>3}-{i+8:<3} ({start_t:>6.1f}-{end_t:<6.1f}s): {label_s} ({avg:.3f})")

    # --- STEM ENERGY ANALYSIS ---
    # Use spectral features as proxy for stem content
    print("\n  [Spectral zones — proxy for stem content]")

    # Low freq energy (bass: 20-250 Hz)
    # Mid freq energy (vocals/other: 250-4000 Hz)
    # High freq energy (hi-hats/cymbals: 4000+ Hz)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000

    # Per 4-bar phrase
    hop_dur = 512 / sr
    for i in range(0, len(bars), 8):
        start_t = bars[i]
        end_t = bars[min(i+8, len(bars)-1)] if i+8 < len(bars) else duration
        start_f = int(start_t / hop_dur)
        end_f = int(end_t / hop_dur)
        end_f = min(end_f, S.shape[1])

        if start_f >= end_f:
            continue

        chunk = S[:, start_f:end_f]
        low_e = np.mean(chunk[low_mask, :])
        mid_e = np.mean(chunk[mid_mask, :])
        high_e = np.mean(chunk[high_mask, :])
        total = low_e + mid_e + high_e + 1e-8

        low_pct = low_e / total * 100
        mid_pct = mid_e / total * 100
        high_pct = high_e / total * 100

        print(f"    Bars {i+1:>3}-{i+8:<3}: Bass {low_pct:4.0f}% | Mid {mid_pct:4.0f}% | Hi {high_pct:4.0f}%")

    # --- ONSET STRENGTH (rhythmic density) ---
    print("\n  [Rhythmic density per 8-bar phrase]")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    for i in range(0, len(bars), 8):
        start_t = bars[i]
        end_t = bars[min(i+8, len(bars)-1)] if i+8 < len(bars) else duration
        start_f = int(start_t / hop_dur)
        end_f = int(end_t / hop_dur)
        end_f = min(end_f, len(onset_env))

        if start_f >= end_f:
            continue

        density = np.mean(onset_env[start_f:end_f])
        visual = "|" * int(density * 2)
        print(f"    Bars {i+1:>3}-{i+8:<3}: {visual:<30} ({density:.1f})")

    return {
        'duration': duration,
        'bpm': round(bpm, 1),
        'key': key_str,
        'bars': bars.tolist(),
        'beats': beats.tolist(),
        'bar_energies': bar_energies_norm,
        'phrases': phrases,
    }


if __name__ == "__main__":
    import sys
    tracks = sys.argv[1:] if len(sys.argv) > 1 else ["track1.mp3", "track2.mp3"]
    results = {}
    for i, t in enumerate(tracks):
        results[f"track_{chr(65+i)}"] = analyze_track(t, f"Track {chr(65+i)}")

    # Summary
    print(f"\n\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k}: {v['bpm']} BPM | {v['key']} | {v['duration']:.1f}s | {len(v['bars'])} bars")
