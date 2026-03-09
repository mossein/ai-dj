#!/usr/bin/env python3
"""
AI DJ Web App — Flask backend + audio engine.
Claude controls this via API calls. User watches the UI.

Endpoints:
  GET  /                  — Web UI
  GET  /api/status        — Current state (playing, position, queue, etc.)
  POST /api/play          — Play a track: {"track": "path.mp3"}
  POST /api/load_next     — Queue next track: {"track": "path.mp3"}
  POST /api/download      — Download from YouTube: {"url": "...", "name": "track3"}
  POST /api/transition    — Mix to next: {"style": "bass_swap|smooth|hard|echo_out|vocal_ride"}
  POST /api/pause
  POST /api/resume
  POST /api/stop
  POST /api/volume        — {"level": 0.8}
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
from flask import Flask, jsonify, request, render_template_string

STEM_CACHE_DIR = "stem_cache"

app = Flask(__name__)


# ============================================================
#  DSP UTILS
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

def load_cached_stems(path, target_sr=None):
    fh = get_file_hash(path)
    d = os.path.join(STEM_CACHE_DIR, fh)
    names = ['drums', 'bass', 'vocals', 'other']
    if not (os.path.isdir(d) and all(os.path.exists(f"{d}/{s}.wav") for s in names)):
        return None
    stems = {}
    for n in names:
        a, fsr = sf.read(f"{d}/{n}.wav", always_2d=True); a = a.T
        if target_sr and fsr != target_sr:
            a = np.stack([librosa.resample(a[ch], orig_sr=fsr, target_sr=target_sr) for ch in range(a.shape[0])])
        stems[n] = a
    return stems

def run_demucs(path):
    fh = get_file_hash(path)
    d = os.path.join(STEM_CACHE_DIR, fh)
    if os.path.isdir(d) and all(os.path.exists(f"{d}/{s}.wav") for s in ['drums','bass','vocals','other']):
        return
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
#  TRACK INFO
# ============================================================
class TrackInfo:
    def __init__(self, path, sr=48000):
        self.path = path
        self.name = os.path.basename(path)
        self.sr = sr
        self.audio = None
        self.stems = None
        self.beats = None
        self.bars = None
        self.bpm = None
        self.lufs = None
        self.duration = 0
        self.ready = False
        self.error = None

    def analyze(self):
        try:
            log(f"Analyzing {self.name}...")
            y, fsr = sf.read(self.path, always_2d=True); y = y.T
            if fsr != self.sr:
                y = np.stack([librosa.resample(y[ch], orig_sr=fsr, target_sr=self.sr)
                              for ch in range(y.shape[0])])
            self.audio = y
            self.duration = y.shape[1] / self.sr
            self.lufs = measure_lufs(y, self.sr)

            log(f"  Detecting beats...")
            self.beats = detect_beats(self.path)
            self.bars = detect_downbeats(self.path)
            self.bpm = compute_bpm(self.beats)

            log(f"  Separating stems...")
            run_demucs(self.path)
            self.stems = load_cached_stems(self.path, target_sr=self.sr)

            self.ready = True
            log(f"  {self.name} ready: {self.bpm} BPM, {self.lufs:.1f} LUFS, {self.duration:.1f}s")
        except Exception as e:
            self.error = str(e)
            log(f"  ERROR analyzing {self.name}: {e}")
            traceback.print_exc()

    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'bpm': self.bpm,
            'lufs': round(self.lufs, 1) if self.lufs and not np.isinf(self.lufs) else None,
            'duration': round(self.duration, 1),
            'ready': self.ready,
            'error': self.error,
        }


# ============================================================
#  TRANSITION BUILDER
# ============================================================
def build_transition(track_a, track_b, style="bass_swap"):
    sr = track_a.sr
    stems_a = track_a.stems
    bars_a = track_a.bars
    bars_b = track_b.bars.copy()

    # Time stretch B to A's BPM
    stems_b = {}
    rb_ratio = track_b.bpm / track_a.bpm
    if abs(rb_ratio - 1.0) > 0.002:
        log(f"  Stretching {track_b.name}: {track_b.bpm} -> {track_a.bpm} BPM")
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
        gain = 10 ** ((track_a.lufs - b_lufs) / 20.0)
        for name in stems_b:
            stems_b[name] = stems_b[name] * gain
        y_b_full = y_b_full * gain

    # Find loop point (energy dip)
    y_mono = np.mean(track_a.audio, axis=0)
    hop = sr // 2
    rms = librosa.feature.rms(y=y_mono.astype(np.float32), frame_length=hop, hop_length=hop)[0]
    rms_norm = rms / (np.max(rms) + 1e-8)
    best_idx, best_score = len(rms_norm)//2, 0
    for i in range(16, len(rms_norm)-16):
        pct = (i*0.5) / track_a.duration
        if not (0.35 < pct < 0.85): continue
        before = np.mean(rms_norm[max(0,i-16):i])
        if before > 0.4 and rms_norm[i] < before*0.6:
            score = before - rms_norm[i]
            if score > best_score:
                best_score, best_idx = score, i
    dip_time = best_idx * 0.5
    dip_bar = np.argmin(np.abs(bars_a - dip_time))
    loop_bars = 8
    loop_start_idx = max(0, dip_bar - loop_bars)
    loop_end_idx = min(loop_start_idx + loop_bars, len(bars_a)-1)
    loop_start = int(bars_a[loop_start_idx] * sr)
    loop_end = int(bars_a[loop_end_idx] * sr)

    # Build loop (3x)
    n_repeats = 3
    xfade = min(int(0.03*sr), (loop_end-loop_start)//4)
    a_looped = {}
    for name in stems_a:
        end = min(loop_end, stems_a[name].shape[1])
        sec = stems_a[name][:, loop_start:end]
        nch, ll = sec.shape
        xf = min(xfade, ll//4); eff = ll-xf; tot = eff*n_repeats+xf
        res = np.zeros((nch, tot))
        for rep in range(n_repeats):
            off = rep*eff; ch = sec.copy()
            if rep > 0: ch[:,:xf] *= np.linspace(0,1,xf)[np.newaxis,:]
            if rep < n_repeats-1: ch[:,-xf:] *= np.linspace(1,0,xf)[np.newaxis,:]
            res[:, off:off+ll] += ch
        a_looped[name] = res

    loop_n = min(s.shape[1] for s in a_looped.values())
    for name in a_looped: a_looped[name] = a_looped[name][:,:loop_n]

    # Track B cue (first energetic section)
    b_cue_bar = 0
    for i in range(0, len(bars_b)-8, 8):
        s0 = int(bars_b[i]*sr); s1 = int(bars_b[min(i+8,len(bars_b)-1)]*sr)
        s1 = min(s1, y_b_full.shape[1])
        if s0 < s1 and np.sqrt(np.mean(y_b_full[:,s0:s1]**2)) > 0.15:
            b_cue_bar = i; break
    b_cue_bar = min(b_cue_bar, len(bars_b)-1)
    b_cue_sample = int(bars_b[b_cue_bar] * sr)

    b_section = {}
    for name in stems_b:
        end = min(b_cue_sample+loop_n, stems_b[name].shape[1])
        seg = stems_b[name][:, b_cue_sample:end]
        if seg.shape[1] < loop_n:
            seg = np.concatenate([seg, np.zeros((seg.shape[0], loop_n-seg.shape[1]))], axis=1)
        b_section[name] = seg[:,:loop_n]

    n = loop_n
    swap_pct = 0.65

    # --- Envelopes by style ---
    if style == "bass_swap":
        env_a = {
            'drums': make_envelope([(0,1),(0.25,1),(0.4,0.8),(swap_pct,0)], n),
            'bass': make_envelope([(0,1),(swap_pct-.01,1),(swap_pct,0)], n),
            'vocals': make_envelope([(0,.8),(.15,.5),(.3,0)], n),
            'other': make_envelope([(0,1),(.2,.8),(.4,.4),(.6,.15),(.75,0)], n),
        }
        env_b = {
            'drums': make_envelope([(0,0),(.05,.12),(.15,.2),(.3,.35),(.5,.6),(swap_pct,.85),(.75,1)], n),
            'bass': make_envelope([(0,0),(swap_pct-.01,0),(swap_pct,1)], n),
            'vocals': make_envelope([(0,0),(.6,0),(swap_pct,0),(.75,.2),(.9,.8),(1,1)], n),
            'other': make_envelope([(0,0),(.3,0),(.4,.1),(.55,.3),(swap_pct,.6),(.8,.9),(.9,1)], n),
        }
        # HP sweep B drums
        hp_end = int(.5*n); nc=32; cs=hp_end//nc; fr=np.linspace(6000,80,nc)
        bd = b_section['drums'][:,:hp_end].copy()
        for i in range(nc):
            s0=i*cs; s1=s0+cs if i<nc-1 else hp_end
            if fr[i]>30: bd[:,s0:s1] = highpass(bd[:,s0:s1],sr,fr[i])
        b_section['drums'][:,:hp_end] = bd

    elif style == "smooth":
        env_a = {s: make_envelope([(0,1),(.3,.9),(.7,.3),(1,0)], n) for s in ['drums','bass','vocals','other']}
        env_b = {s: make_envelope([(0,0),(.3,.1),(.7,.7),(1,1)], n) for s in ['drums','bass','vocals','other']}

    elif style == "hard":
        cut = 0.3
        env_a = {s: make_envelope([(0,1),(cut-.02,1),(cut,0)], n) for s in ['drums','bass','vocals','other']}
        env_b = {s: make_envelope([(0,0),(cut-.02,0),(cut,1)], n) for s in ['drums','bass','vocals','other']}

    elif style == "echo_out":
        env_a = {
            'drums': make_envelope([(0,1),(.3,.5),(.5,0)], n),
            'bass': make_envelope([(0,1),(.4,.8),(.6,0)], n),
            'vocals': make_envelope([(0,1),(.2,.7),(.5,0)], n),
            'other': make_envelope([(0,1),(.3,.8),(.7,.2),(1,0)], n),
        }
        env_b = {
            'drums': make_envelope([(0,0),(.2,.2),(.5,.7),(.7,1)], n),
            'bass': make_envelope([(0,0),(.4,0),(.5,.5),(.7,1)], n),
            'vocals': make_envelope([(0,0),(.5,0),(.7,.5),(.9,1)], n),
            'other': make_envelope([(0,0),(.3,.1),(.5,.4),(.7,.8),(.9,1)], n),
        }

    elif style == "vocal_ride":
        env_a = {
            'drums': make_envelope([(0,1),(.2,.7),(.4,.3),(.55,0)], n),
            'bass': make_envelope([(0,1),(.3,.8),(swap_pct-.01,.5),(swap_pct,0)], n),
            'vocals': make_envelope([(0,1),(.4,1),(.6,.9),(.8,.7),(.9,.3),(1,0)], n),
            'other': make_envelope([(0,1),(.25,.6),(.5,.2),(.7,0)], n),
        }
        env_b = {
            'drums': make_envelope([(0,0),(.1,.2),(.25,.5),(.4,.8),(.55,1)], n),
            'bass': make_envelope([(0,0),(swap_pct-.01,0),(swap_pct,1)], n),
            'vocals': make_envelope([(0,0),(.8,0),(.9,.5),(1,1)], n),
            'other': make_envelope([(0,0),(.15,.1),(.3,.3),(.5,.6),(.7,.9),(.8,1)], n),
        }
        hp_end = int(.4*n); nc=32; cs=hp_end//nc; fr=np.linspace(6000,80,nc)
        bd = b_section['drums'][:,:hp_end].copy()
        for i in range(nc):
            s0=i*cs; s1=s0+cs if i<nc-1 else hp_end
            if fr[i]>30: bd[:,s0:s1] = highpass(bd[:,s0:s1],sr,fr[i])
        b_section['drums'][:,:hp_end] = bd

    else:
        # Default to bass_swap
        return build_transition(track_a, track_b, style="bass_swap")

    # Reverb wash on A other
    rs = int(.2*n); re = int(.75*n)
    board = Pedalboard([HighpassFilter(cutoff_frequency_hz=250),
                        Reverb(room_size=0.85,damping=0.6,wet_level=0.4,dry_level=0.0),
                        LowpassFilter(cutoff_frequency_hz=6000)])
    a_rev = board(a_looped['other'][:,rs:re].copy().astype(np.float32), sr)

    # Mix transition
    transition = np.zeros((2, n))
    for stem in ['drums','bass','vocals','other']:
        ac = a_looped[stem] * env_a[stem][np.newaxis,:]
        bc = b_section[stem] * env_b[stem][np.newaxis,:]
        if stem == 'other':
            rl = min(a_rev.shape[1], re-rs)
            if rl > 0:
                rb = np.linspace(0,.6,rl); db = np.linspace(1,.4,rl)
                ac[:,rs:rs+rl] = ac[:,rs:rs+rl]*db[np.newaxis,:] + a_rev[:,:rl]*rb[np.newaxis,:]*env_a['other'][rs:rs+rl][np.newaxis,:]
        transition += ac + bc

    pk = np.max(np.abs(transition))
    if pk > 0.95:
        sc = 1.2/pk
        transition = np.tanh(transition*sc)/np.tanh(sc)

    a_before = track_a.audio[:, :loop_start]
    b_cont = b_cue_sample + loop_n
    b_after = y_b_full[:, b_cont:] if b_cont < y_b_full.shape[1] else np.zeros((2, sr))

    # Anti-click crossfades
    xfl = min(int(0.005*sr), 512)
    if xfl > 0 and a_before.shape[1] > xfl and transition.shape[1] > xfl:
        fo=np.linspace(1,0,xfl)[np.newaxis,:]; fi=np.linspace(0,1,xfl)[np.newaxis,:]
        bl = a_before[:,-xfl:]*fo + transition[:,:xfl]*fi
        a_before = np.concatenate([a_before[:,:-xfl], bl], axis=1)
        transition = transition[:,xfl:]
    if xfl > 0 and transition.shape[1] > xfl and b_after.shape[1] > xfl:
        fo=np.linspace(1,0,xfl)[np.newaxis,:]; fi=np.linspace(0,1,xfl)[np.newaxis,:]
        bl = transition[:,-xfl:]*fo + b_after[:,:xfl]*fi
        transition = np.concatenate([transition[:,:-xfl], bl], axis=1)
        b_after = b_after[:,xfl:]

    full = np.clip(np.concatenate([a_before, transition, b_after], axis=1), -0.98, 0.98)

    return {
        'buffer': full,
        'trans_start': a_before.shape[1],
        'trans_end': a_before.shape[1] + transition.shape[1],
        'style': style,
        'loop_bars': f"{loop_start_idx+1}-{loop_end_idx}",
        'b_cue_bar': b_cue_bar + 1,
    }


# ============================================================
#  DJ ENGINE
# ============================================================
class DJEngine:
    def __init__(self):
        self.sr = 48000
        self.lock = threading.Lock()
        self.buffer = None
        self.position = 0
        self.playing = False
        self.paused = False
        self.volume = 0.9
        self.current_track = None
        self.next_track = None
        self.next_analyzing = False
        self.next_ready = False
        self.transition_active = False
        self.transition_building = False
        self.trans_start = 0
        self.trans_end = 0
        self.logs = []
        self.stream = None
        self.downloading = False
        self.download_name = None

    def add_log(self, msg):
        self.logs.append({'time': time.time(), 'msg': msg})
        if len(self.logs) > 50: self.logs = self.logs[-50:]
        print(f"  [DJ] {msg}")

    def audio_callback(self, outdata, frames, time_info, status):
        with self.lock:
            if self.buffer is None or self.paused or not self.playing:
                outdata[:] = 0; return
            pos = self.position; total = self.buffer.shape[1]
            end = min(pos+frames, total); nf = end-pos
            if nf <= 0:
                outdata[:] = 0; self.playing = False; return
            outdata[:nf] = (self.buffer[:,pos:end]*self.volume).T.astype(np.float32)
            if nf < frames: outdata[nf:] = 0
            self.position = end

    def start_stream(self):
        if self.stream is None:
            self.stream = sd.OutputStream(samplerate=self.sr, channels=2,
                                          blocksize=2048, callback=self.audio_callback, dtype='float32')
            self.stream.start()

    def play_track(self, path):
        self.add_log(f"Loading {os.path.basename(path)}...")
        info = TrackInfo(path, sr=self.sr)
        info.analyze()
        if not info.ready:
            self.add_log(f"Failed to load: {info.error}")
            return False
        with self.lock:
            self.buffer = info.audio.copy()
            self.position = 0
            self.playing = True
            self.paused = False
            self.current_track = info
            self.transition_active = False
        self.start_stream()
        self.add_log(f"Now playing: {info.name} ({info.bpm} BPM, {info.duration:.0f}s)")
        return True

    def load_next(self, path):
        self.next_analyzing = True
        self.next_ready = False
        def _do():
            try:
                info = TrackInfo(path, sr=self.sr)
                info.analyze()
                self.next_track = info
                self.next_ready = info.ready
                self.add_log(f"Next ready: {info.name} ({info.bpm} BPM)")
            except Exception as e:
                self.add_log(f"Failed to load next: {e}")
            finally:
                self.next_analyzing = False
        threading.Thread(target=_do, daemon=True).start()

    def do_transition(self, style="bass_swap"):
        if not self.next_ready or not self.next_track:
            self.add_log("No next track ready!")
            return False
        self.transition_building = True
        def _do():
            try:
                self.add_log(f"Building transition: {style}...")
                result = build_transition(self.current_track, self.next_track, style=style)
                with self.lock:
                    self.buffer = result['buffer']
                    self.position = 0
                    self.playing = True
                    self.transition_active = True
                    self.trans_start = result['trans_start']
                    self.trans_end = result['trans_end']
                self.start_stream()
                self.current_track = self.next_track
                self.next_track = None
                self.next_ready = False
                self.add_log(f"Transition fired! Style: {style}, loop bars: {result['loop_bars']}, B cue: bar {result['b_cue_bar']}")
            except Exception as e:
                self.add_log(f"Transition failed: {e}")
                traceback.print_exc()
            finally:
                self.transition_building = False
        threading.Thread(target=_do, daemon=True).start()
        return True

    def download_track(self, url, name):
        self.downloading = True
        self.download_name = name
        def _do():
            try:
                import yt_dlp
                path = f"{name}.mp3"
                self.add_log(f"Downloading: {url}")
                opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{'key': 'FFmpegExtractAudio',
                                       'preferredcodec': 'mp3', 'preferredquality': '192'}],
                    'outtmpl': name, 'quiet': True,
                }
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
                self.add_log(f"Downloaded: {path}")
                # Auto-queue as next
                self.load_next(path)
            except Exception as e:
                self.add_log(f"Download failed: {e}")
            finally:
                self.downloading = False
        threading.Thread(target=_do, daemon=True).start()

    def get_status(self):
        pos = self.position
        total = self.buffer.shape[1] if self.buffer is not None else 0
        return {
            'playing': self.playing,
            'paused': self.paused,
            'position_s': round(pos / self.sr, 1) if total else 0,
            'duration_s': round(total / self.sr, 1) if total else 0,
            'progress_pct': round(pos / total * 100, 1) if total else 0,
            'volume': self.volume,
            'current_track': self.current_track.to_dict() if self.current_track else None,
            'next_track': self.next_track.to_dict() if self.next_track else None,
            'next_analyzing': self.next_analyzing,
            'next_ready': self.next_ready,
            'transition_active': self.transition_active,
            'transition_building': self.transition_building,
            'trans_zone': {
                'start_s': round(self.trans_start/self.sr, 1),
                'end_s': round(self.trans_end/self.sr, 1),
            } if self.transition_active else None,
            'downloading': self.downloading,
            'download_name': self.download_name,
            'logs': [l['msg'] for l in self.logs[-15:]],
        }


# Global engine
engine = DJEngine()

def log(msg):
    engine.add_log(msg)


# ============================================================
#  API ROUTES
# ============================================================
@app.route('/api/status')
def api_status():
    return jsonify(engine.get_status())

@app.route('/api/play', methods=['POST'])
def api_play():
    track = request.json.get('track', '')
    if not os.path.exists(track):
        return jsonify({'error': f'File not found: {track}'}), 404
    threading.Thread(target=engine.play_track, args=(track,), daemon=True).start()
    return jsonify({'ok': True, 'track': track})

@app.route('/api/load_next', methods=['POST'])
def api_load_next():
    track = request.json.get('track', '')
    if not os.path.exists(track):
        return jsonify({'error': f'File not found: {track}'}), 404
    engine.load_next(track)
    return jsonify({'ok': True, 'track': track})

@app.route('/api/download', methods=['POST'])
def api_download():
    url = request.json.get('url', '')
    name = request.json.get('name', 'track_dl')
    if not url:
        return jsonify({'error': 'No URL'}), 400
    engine.download_track(url, name)
    return jsonify({'ok': True, 'url': url, 'name': name})

@app.route('/api/transition', methods=['POST'])
def api_transition():
    style = request.json.get('style', 'bass_swap') if request.json else 'bass_swap'
    ok = engine.do_transition(style=style)
    return jsonify({'ok': ok, 'style': style})

@app.route('/api/pause', methods=['POST'])
def api_pause():
    engine.paused = True
    engine.add_log("Paused")
    return jsonify({'ok': True})

@app.route('/api/resume', methods=['POST'])
def api_resume():
    engine.paused = False
    engine.add_log("Resumed")
    return jsonify({'ok': True})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    engine.playing = False
    engine.add_log("Stopped")
    return jsonify({'ok': True})

@app.route('/api/volume', methods=['POST'])
def api_volume():
    level = float(request.json.get('level', 0.9))
    engine.volume = max(0, min(1, level))
    return jsonify({'ok': True, 'volume': engine.volume})

@app.route('/api/tracks')
def api_tracks():
    """List available mp3 files."""
    files = sorted([f for f in os.listdir('.') if f.endswith('.mp3')])
    return jsonify({'tracks': files})


# ============================================================
#  WEB UI
# ============================================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI DJ — Claude's Decks</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a0f; color: #e0e0e0; font-family: 'SF Mono', 'Fira Code', monospace; min-height: 100vh; }

  .container { max-width: 900px; margin: 0 auto; padding: 20px; }

  h1 { text-align: center; font-size: 2em; margin-bottom: 5px;
       background: linear-gradient(135deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 0.85em; }

  .card { background: #151520; border: 1px solid #2a2a3a; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .card h2 { color: #48dbfb; font-size: 1em; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 2px; }

  /* Now Playing */
  .track-name { font-size: 1.4em; color: #fff; margin-bottom: 4px; }
  .track-info { color: #888; font-size: 0.85em; margin-bottom: 12px; }

  .progress-bar { width: 100%; height: 8px; background: #2a2a3a; border-radius: 4px; overflow: hidden; margin-bottom: 8px; position: relative; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  .progress-fill.normal { background: linear-gradient(90deg, #48dbfb, #6c5ce7); }
  .progress-fill.transition { background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb); }
  .time-display { display: flex; justify-content: space-between; color: #888; font-size: 0.8em; }

  /* Buttons */
  .btn-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
  .btn { padding: 10px 20px; border: 1px solid #3a3a4a; border-radius: 8px; background: #1a1a2a;
         color: #e0e0e0; cursor: pointer; font-family: inherit; font-size: 0.85em; transition: all 0.2s; }
  .btn:hover { background: #2a2a4a; border-color: #48dbfb; }
  .btn:active { transform: scale(0.97); }
  .btn.primary { background: #6c5ce7; border-color: #6c5ce7; color: #fff; }
  .btn.primary:hover { background: #7d6df0; }
  .btn.danger { border-color: #ff6b6b; color: #ff6b6b; }
  .btn.danger:hover { background: #2a1a1a; }
  .btn.success { border-color: #00d2d3; color: #00d2d3; }
  .btn.success:hover { background: #1a2a2a; }
  .btn:disabled { opacity: 0.3; cursor: not-allowed; }

  /* Transition buttons */
  .trans-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .trans-btn { padding: 12px 16px; border: 2px solid #ff6b6b; border-radius: 10px; background: #1a1015;
               color: #ff6b6b; cursor: pointer; font-family: inherit; font-size: 0.9em; font-weight: bold;
               transition: all 0.2s; flex: 1; min-width: 120px; text-align: center; }
  .trans-btn:hover { background: #2a1520; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(255,107,107,0.2); }
  .trans-btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
  .trans-btn.vocal { border-color: #feca57; color: #feca57; background: #1a1810; }
  .trans-btn.vocal:hover { background: #2a2815; box-shadow: 0 4px 15px rgba(254,202,87,0.2); }
  .trans-btn.smooth { border-color: #48dbfb; color: #48dbfb; background: #101520; }
  .trans-btn.smooth:hover { background: #152530; box-shadow: 0 4px 15px rgba(72,219,251,0.2); }
  .trans-btn.echo { border-color: #ff9ff3; color: #ff9ff3; background: #1a1018; }
  .trans-btn.echo:hover { background: #2a1528; box-shadow: 0 4px 15px rgba(255,159,243,0.2); }
  .trans-btn.hard { border-color: #ff4444; color: #ff4444; background: #1a0f0f; }
  .trans-btn.hard:hover { background: #2a1515; box-shadow: 0 4px 15px rgba(255,68,68,0.2); }

  /* Next track */
  .next-status { padding: 8px 12px; border-radius: 6px; font-size: 0.85em; }
  .next-status.ready { background: #0a2a1a; color: #00d2d3; border: 1px solid #00d2d3; }
  .next-status.analyzing { background: #2a2a0a; color: #feca57; border: 1px solid #feca57; }
  .next-status.none { background: #1a1a1a; color: #666; border: 1px solid #333; }

  /* Track list */
  .track-list { display: flex; flex-direction: column; gap: 6px; }
  .track-item { display: flex; justify-content: space-between; align-items: center;
                padding: 8px 12px; background: #1a1a25; border-radius: 6px; }
  .track-item:hover { background: #22223a; }
  .track-item .name { flex: 1; }

  /* YouTube input */
  .yt-row { display: flex; gap: 10px; }
  .yt-input { flex: 1; padding: 10px; background: #1a1a2a; border: 1px solid #3a3a4a; border-radius: 8px;
              color: #e0e0e0; font-family: inherit; font-size: 0.85em; }
  .yt-input:focus { outline: none; border-color: #ff6b6b; }
  .yt-input::placeholder { color: #555; }

  /* Volume */
  .volume-slider { width: 100%; -webkit-appearance: none; height: 6px; border-radius: 3px;
                   background: #2a2a3a; outline: none; }
  .volume-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px;
                                         border-radius: 50%; background: #48dbfb; cursor: pointer; }

  /* Log */
  .log { max-height: 200px; overflow-y: auto; font-size: 0.75em; color: #666; line-height: 1.6; }
  .log .entry { border-bottom: 1px solid #1a1a25; padding: 2px 0; }

  /* Status indicator */
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .status-dot.playing { background: #00d2d3; box-shadow: 0 0 8px #00d2d3; animation: pulse 2s infinite; }
  .status-dot.paused { background: #feca57; }
  .status-dot.stopped { background: #666; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }

  .building { animation: buildPulse 1s infinite; }
  @keyframes buildPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
</head>
<body>
<div class="container">
  <h1>AI DJ</h1>
  <p class="subtitle">Claude's Decks — Talk to Claude to control the mix</p>

  <!-- Now Playing -->
  <div class="card">
    <h2><span class="status-dot" id="statusDot"></span> Now Playing</h2>
    <div class="track-name" id="trackName">—</div>
    <div class="track-info" id="trackInfo">No track loaded</div>
    <div class="progress-bar"><div class="progress-fill normal" id="progressFill"></div></div>
    <div class="time-display">
      <span id="timePos">0:00</span>
      <span id="transLabel"></span>
      <span id="timeDur">0:00</span>
    </div>
    <div class="btn-row">
      <button class="btn" onclick="apiPost('/api/pause')">Pause</button>
      <button class="btn" onclick="apiPost('/api/resume')">Resume</button>
      <button class="btn danger" onclick="apiPost('/api/stop')">Stop</button>
    </div>
  </div>

  <!-- Transition Controls -->
  <div class="card">
    <h2>Transition</h2>
    <div id="nextStatus" class="next-status none" style="margin-bottom:12px;">No next track queued</div>
    <div class="trans-row">
      <button class="trans-btn" id="btnBassSwap" onclick="doTransition('bass_swap')" disabled>Bass Swap</button>
      <button class="trans-btn vocal" id="btnVocal" onclick="doTransition('vocal_ride')" disabled>Vocal Ride</button>
      <button class="trans-btn smooth" id="btnSmooth" onclick="doTransition('smooth')" disabled>Smooth</button>
      <button class="trans-btn echo" id="btnEcho" onclick="doTransition('echo_out')" disabled>Echo Out</button>
      <button class="trans-btn hard" id="btnHard" onclick="doTransition('hard')" disabled>Hard Cut</button>
    </div>
    <div id="buildingMsg" style="margin-top:8px;display:none;" class="building">Building transition...</div>
  </div>

  <!-- Load Track -->
  <div class="card">
    <h2>Load Track</h2>
    <div class="track-list" id="trackList">Loading...</div>
  </div>

  <!-- YouTube Download -->
  <div class="card">
    <h2>Download from YouTube</h2>
    <div class="yt-row">
      <input class="yt-input" id="ytUrl" placeholder="Paste YouTube URL..." />
      <input class="yt-input" id="ytName" placeholder="Name" style="max-width:150px;" value="track_new" />
      <button class="btn primary" onclick="doDownload()" id="dlBtn">Download</button>
    </div>
    <div id="dlStatus" style="margin-top:8px;font-size:0.8em;color:#888;"></div>
  </div>

  <!-- Volume -->
  <div class="card">
    <h2>Volume: <span id="volLabel">90%</span></h2>
    <input class="volume-slider" type="range" min="0" max="100" value="90" oninput="setVolume(this.value)" />
  </div>

  <!-- Log -->
  <div class="card">
    <h2>DJ Log</h2>
    <div class="log" id="logBox"></div>
  </div>
</div>

<script>
function fmt(s) { const m=Math.floor(s/60); return m+':'+String(Math.floor(s%60)).padStart(2,'0'); }

function apiPost(url, body) {
  fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: body ? JSON.stringify(body) : '{}'});
}

function doTransition(style) {
  apiPost('/api/transition', {style});
}

function doDownload() {
  const url = document.getElementById('ytUrl').value;
  const name = document.getElementById('ytName').value || 'track_new';
  if (!url) return;
  apiPost('/api/download', {url, name});
  document.getElementById('dlBtn').disabled = true;
  document.getElementById('dlStatus').textContent = 'Downloading...';
}

function setVolume(v) {
  document.getElementById('volLabel').textContent = v+'%';
  apiPost('/api/volume', {level: v/100});
}

function loadTrackAs(type, file) {
  if (type === 'play') apiPost('/api/play', {track: file});
  else apiPost('/api/load_next', {track: file});
}

// Fetch track list
fetch('/api/tracks').then(r=>r.json()).then(d => {
  const el = document.getElementById('trackList');
  el.innerHTML = '';
  d.tracks.forEach(f => {
    const div = document.createElement('div');
    div.className = 'track-item';
    div.innerHTML = `<span class="name">${f}</span>
      <button class="btn" style="padding:4px 10px;font-size:0.8em;" onclick="loadTrackAs('play','${f}')">Play</button>
      <button class="btn success" style="padding:4px 10px;font-size:0.8em;margin-left:4px;" onclick="loadTrackAs('next','${f}')">Queue Next</button>`;
    el.appendChild(div);
  });
});

// Poll status
setInterval(() => {
  fetch('/api/status').then(r=>r.json()).then(s => {
    // Status dot
    const dot = document.getElementById('statusDot');
    dot.className = 'status-dot ' + (s.paused?'paused':s.playing?'playing':'stopped');

    // Track info
    const ct = s.current_track;
    document.getElementById('trackName').textContent = ct ? ct.name : '—';
    document.getElementById('trackInfo').textContent = ct
      ? `${ct.bpm} BPM · ${ct.lufs} LUFS · ${fmt(ct.duration)}`
      : 'No track loaded';

    // Progress
    const fill = document.getElementById('progressFill');
    fill.style.width = s.progress_pct + '%';
    fill.className = 'progress-fill ' + (s.transition_active && s.trans_zone
      && s.position_s >= s.trans_zone.start_s && s.position_s <= s.trans_zone.end_s
      ? 'transition' : 'normal');

    document.getElementById('timePos').textContent = fmt(s.position_s);
    document.getElementById('timeDur').textContent = fmt(s.duration_s);

    // Transition label
    const tl = document.getElementById('transLabel');
    if (s.transition_active && s.trans_zone) {
      if (s.position_s >= s.trans_zone.start_s && s.position_s <= s.trans_zone.end_s)
        tl.textContent = 'TRANSITIONING';
      else if (s.position_s < s.trans_zone.start_s)
        tl.textContent = `Transition in ${fmt(s.trans_zone.start_s - s.position_s)}`;
      else tl.textContent = '';
    } else tl.textContent = '';

    // Next track status
    const ns = document.getElementById('nextStatus');
    const nt = s.next_track;
    const transBtns = ['btnBassSwap','btnVocal','btnSmooth','btnEcho','btnHard'];
    if (s.next_ready && nt) {
      ns.className = 'next-status ready';
      ns.textContent = `Ready: ${nt.name} (${nt.bpm} BPM)`;
      transBtns.forEach(id => document.getElementById(id).disabled = false);
    } else if (s.next_analyzing) {
      ns.className = 'next-status analyzing';
      ns.textContent = 'Analyzing next track...';
      transBtns.forEach(id => document.getElementById(id).disabled = true);
    } else {
      ns.className = 'next-status none';
      ns.textContent = 'No next track queued';
      transBtns.forEach(id => document.getElementById(id).disabled = true);
    }

    // Building
    document.getElementById('buildingMsg').style.display = s.transition_building ? 'block' : 'none';

    // Download
    if (!s.downloading) {
      document.getElementById('dlBtn').disabled = false;
      if (document.getElementById('dlStatus').textContent === 'Downloading...')
        document.getElementById('dlStatus').textContent = 'Done!';
    }

    // Logs
    const lb = document.getElementById('logBox');
    lb.innerHTML = s.logs.map(l => `<div class="entry">${l}</div>`).join('');
    lb.scrollTop = lb.scrollHeight;
  }).catch(()=>{});
}, 500);
</script>
</body>
</html>
"""


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5555)
    parser.add_argument('--play', type=str, default=None, help='Auto-play this track on start')
    args = parser.parse_args()

    if args.play:
        threading.Thread(target=engine.play_track, args=(args.play,), daemon=True).start()

    print(f"\n  DJ App running at http://localhost:{args.port}")
    print(f"  Claude controls via API, you watch the UI\n")
    app.run(host='0.0.0.0', port=args.port, debug=False, use_reloader=False)
