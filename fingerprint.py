"""
Shazam-like audio fingerprinting.

Pipeline (Wang, 2003):
    audio (time series)  --STFT-->  spectrogram
    spectrogram          --peaks->  constellation map
    constellation map    --pairs->  landmark hashes (anchor, target, dt)
    hashes + song_id     --index->  inverted hash table
    query hashes         --vote -->  song_id with consistent time offset
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
import pickle

import numpy as np
import librosa
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure


# --- STFT parameters (shared across index and query) ---
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512   # ~23 ms per frame at 22050 Hz
WIN_LENGTH = N_FFT

# --- Peak picking parameters ---
PEAK_NEIGH_FREQ = 20      # half-size of local-max neighborhood (frequency bins)
PEAK_NEIGH_TIME = 20      # half-size of local-max neighborhood (time frames)
PEAK_MIN_DB = -40.0       # peaks below this dB level (relative to max=0) are dropped

# --- Landmark (pair) parameters ---
FANOUT = 10               # number of target peaks paired with each anchor
TARGET_DT_MIN = 1         # frames; minimum time gap anchor->target
TARGET_DT_MAX = 100       # frames; ~2.3 s window
TARGET_DF_MAX = 200       # frequency bins; max |f2 - f1|


# ----------------------------------------------------------------------
# Audio -> spectrogram
# ----------------------------------------------------------------------
def load_audio(path, sr=SAMPLE_RATE, duration=None, offset=0.0):
    """Load mono audio, resampled to `sr`."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration, offset=offset)
    return y


def compute_spectrogram(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    Returns the magnitude STFT in dB.
    Shape: (n_freq_bins, n_time_frames). Reference is the max so 0 dB == loudest bin.
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db


# ----------------------------------------------------------------------
# Spectrogram -> peaks (constellation map)
# ----------------------------------------------------------------------
def find_peaks(S_db,
               neigh_freq=PEAK_NEIGH_FREQ,
               neigh_time=PEAK_NEIGH_TIME,
               min_db=PEAK_MIN_DB):
    """
    Detect local maxima of the spectrogram.

    A bin (f, t) is a peak iff it equals the max within a (2*neigh_freq+1) x
    (2*neigh_time+1) rectangle around it AND its dB value is above `min_db`.

    Returns an (N, 2) array of (frequency_bin, time_frame) pairs.
    """
    size = (2 * neigh_freq + 1, 2 * neigh_time + 1)
    local_max = maximum_filter(S_db, size=size, mode="constant", cval=-np.inf)
    is_peak = (S_db == local_max) & (S_db > min_db)
    f_idx, t_idx = np.where(is_peak)
    peaks = np.stack([f_idx, t_idx], axis=1)
    return peaks


# ----------------------------------------------------------------------
# Peaks -> hashes (combinatorial landmarks)
# ----------------------------------------------------------------------
def _hash_pair(f1: int, f2: int, dt: int) -> int:
    """Pack (f1, f2, dt) into a single 32-bit-ish integer."""
    # f1, f2 fit in 10 bits each (n_fft//2+1 = 1025 bins). dt fits in 12 bits.
    return (int(f1) & 0x3FF) << 22 | (int(f2) & 0x3FF) << 12 | (int(dt) & 0xFFF)


def make_hashes(peaks,
                fanout=FANOUT,
                dt_min=TARGET_DT_MIN,
                dt_max=TARGET_DT_MAX,
                df_max=TARGET_DF_MAX):
    """
    Pair each anchor peak with up to `fanout` future peaks inside a target zone,
    then encode every pair as a hash.

    Returns a list of (hash, anchor_time_frame).
    """
    # sort by time, then frequency, so "future" target peaks come after
    peaks = peaks[np.lexsort((peaks[:, 0], peaks[:, 1]))]
    times = peaks[:, 1]
    freqs = peaks[:, 0]
    n = len(peaks)

    out = []
    j_start = 0
    for i in range(n):
        t1 = times[i]
        f1 = freqs[i]
        # advance window start
        while j_start < n and times[j_start] - t1 < dt_min:
            j_start += 1
        paired = 0
        for j in range(j_start, n):
            dt = int(times[j] - t1)
            if dt > dt_max:
                break
            if dt < dt_min:
                continue
            f2 = freqs[j]
            if abs(int(f2) - int(f1)) > df_max:
                continue
            out.append((_hash_pair(f1, f2, dt), int(t1)))
            paired += 1
            if paired >= fanout:
                break
    return out


def fingerprint_audio(y, **kwargs):
    """Convenience: audio time series -> list of (hash, anchor_t)."""
    S_db = compute_spectrogram(y)
    peaks = find_peaks(S_db,
                       neigh_freq=kwargs.get("neigh_freq", PEAK_NEIGH_FREQ),
                       neigh_time=kwargs.get("neigh_time", PEAK_NEIGH_TIME),
                       min_db=kwargs.get("min_db", PEAK_MIN_DB))
    hashes = make_hashes(peaks,
                         fanout=kwargs.get("fanout", FANOUT))
    return S_db, peaks, hashes


# ----------------------------------------------------------------------
# Catalog index
# ----------------------------------------------------------------------
@dataclass
class Catalog:
    # hash -> list of (song_id, anchor_time_frame)
    index: dict
    # song_id -> human-readable name (file path)
    songs: dict

    def num_hashes(self) -> int:
        return sum(len(v) for v in self.index.values())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"index": dict(self.index), "songs": self.songs}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(index=defaultdict(list, d["index"]), songs=d["songs"])


def build_catalog(file_paths, verbose=True, max_seconds=None):
    """Fingerprint a list of files into a Catalog."""
    index = defaultdict(list)
    songs = {}
    iterator = enumerate(file_paths)
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), total=len(file_paths), desc="Indexing")
        except Exception:
            pass
    for sid, path in iterator:
        try:
            y = load_audio(path, duration=max_seconds)
        except Exception as e:
            if verbose:
                print(f"  skip {path}: {e}")
            continue
        _, _, hashes = fingerprint_audio(y)
        for h, t in hashes:
            index[h].append((sid, t))
        songs[sid] = str(path)
    return Catalog(index=index, songs=songs)


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------
def match(query_audio, catalog: Catalog, top_k=5):
    """
    Recognize a query.

    Algorithm:
      1) Fingerprint the query.
      2) For every query hash, look up matches in the catalog.
      3) For each match, compute offset = t_db - t_query.
      4) For each song, the most frequent offset is the alignment vote.
         The song with the largest vote wins.
    """
    _, _, q_hashes = fingerprint_audio(query_audio)

    # song_id -> Counter of offset -> count
    votes = defaultdict(Counter)
    n_lookups = 0
    n_hits = 0
    for h, tq in q_hashes:
        matches = catalog.index.get(h)
        if not matches:
            continue
        n_lookups += 1
        for sid, td in matches:
            votes[sid][td - tq] += 1
            n_hits += 1

    # Score each song by the height of its tallest offset bin
    scored = []
    for sid, offset_counter in votes.items():
        offset, count = offset_counter.most_common(1)[0]
        scored.append((sid, count, offset, sum(offset_counter.values())))
    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "query_hashes": len(q_hashes),
        "query_hash_hits": n_hits,
        "ranking": scored[:top_k],   # list of (song_id, best_offset_count, best_offset, total_matches)
        "votes": votes,              # full debug info
    }


# ----------------------------------------------------------------------
# Conveniences
# ----------------------------------------------------------------------
def frames_to_seconds(t_frames):
    return np.asarray(t_frames) * HOP_LENGTH / SAMPLE_RATE


def freq_bin_to_hz(f_bin):
    return np.asarray(f_bin) * SAMPLE_RATE / N_FFT


def list_dataset(root):
    """Return all .mp3 paths in fma_small, sorted."""
    return sorted(str(p) for p in Path(root).glob("*/*.mp3"))
