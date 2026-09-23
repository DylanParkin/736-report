"""
Ensemble respiration-rate estimation for the 4 chest-strap respiration
sections used in the report (S7 sitting, S7 walking, S8 sitting, S8 walking -
see make_plots.py), producing a per-window rate robust enough to stand in as
a trustworthy reference rather than a single noisy method.

Per Charlton et al.'s RRest benchmarking findings, no single breath-detection
algorithm is reliable on its own, so each window is scored by three
independent methods and fused by a quality-weighted combination - windows the
methods can't agree on are marked low-confidence (excluded) rather than
reported as a number.

Pipeline
  1. Zero-phase Butterworth band-pass, 0.10-0.50 Hz (6-30 breaths/min), via
     sosfiltfilt over the whole section - no time-shift, at the cost of
     needing the full section in hand. This is deliberately NOT causal /
     real-time; the goal here is a trustworthy reference rate, not a live
     read-out.
  2. Per 8 s window (2 s shift - the same convention as the reference-HR
     windows in make_plots.py), run 3 independent breath-rate estimators:
       - peak     : find_peaks on the filtered waveform, rate from the mean
                    inter-peak interval
       - zero_x   : positive-going zero crossings, rate from the mean
                    inter-crossing interval
       - spectral : zero-padded FFT of the window restricted to the
                    breathing band, rate from the dominant frequency
     each with a [0, 1] quality score - inverse breath-interval CV for the
     two time-domain methods, spectral dominance ratio for the FFT one.
  3. Fuse the window's valid estimates with a quality-weighted average.
  4. Reject the window (NaN) if fewer than 2 methods produced a usable
     estimate, if their spread exceeds AGREEMENT_MAX_BPM, or if the fused
     quality is below QUALITY_MIN.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, savgol_filter, sosfiltfilt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "S7_S8_data", "PPG_FieldStudy")
OUT_DIR = BASE_DIR

SUBJECTS = ["S7", "S8"]
ACTIVITY_CODE = {"sitting": 1, "walking": 7}  # BASELINE, WALKING (readme Sec. II)

FS_CHEST = 700.0     # RespiBAN chest unit sample rate
FS_ACTIVITY = 4.0    # activity-code signal, synced to RespiBAN start time

BAND_LOW_HZ, BAND_HIGH_HZ = 0.10, 0.50           # 6-30 breaths/min
BAND_LOW_BPM, BAND_HIGH_BPM = BAND_LOW_HZ * 60, BAND_HIGH_HZ * 60

WINDOW_S = 8.0        # matches the reference-HR window in make_plots.py
SHIFT_S = 2.0         # matches the reference-HR shift in make_plots.py

MIN_METHODS = 2               # need at least this many methods to vote
QUALITY_MIN = 0.30            # minimum fused (mean) quality to report a window
AGREEMENT_MAX_BPM = 6.0       # max spread allowed across contributing methods
METHOD_QUALITY_FLOOR = 0.15   # a single method below this doesn't get to vote
FFT_ZERO_PAD_S = 60.0         # zero-pad each window's FFT for a finer frequency grid

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"
SERIES_AMBER = "#c8720a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": GRIDLINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "savefig.facecolor": "#fcfcfb",
})


def load_subject(name):
    path = os.path.join(DATA_DIR, name, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def activity_segments(activity_1d, fs):
    """Collapse the per-sample activity code array into contiguous
    (code, start_s, end_s) run-length segments."""
    changes = np.flatnonzero(np.diff(activity_1d)) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(activity_1d)]))
    return [(int(activity_1d[s]), s / fs, e / fs) for s, e in zip(starts, ends)]


def section_window(data, code):
    """Absolute [start_s, end_s) of the (single) activity segment with the
    given code, in RespiBAN/chest time."""
    segments = activity_segments(data["activity"].ravel(), FS_ACTIVITY)
    matches = [(s, e) for c, s, e in segments if c == code]
    return max(matches, key=lambda se: se[1] - se[0])  # longest run of that code


def bandpass_filtfilt(x, fs):
    sos = butter(3, [BAND_LOW_HZ, BAND_HIGH_HZ], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def peak_method(window, fs):
    """Time-domain peak detection: rate from the mean inter-peak interval,
    quality from how regular those intervals are (1 / (1 + CV))."""
    min_distance = max(int(fs * 60.0 / BAND_HIGH_BPM * 0.8), 1)
    prominence = 0.25 * (window.max() - window.min())
    peaks, _ = find_peaks(
        window, distance=min_distance,
        prominence=prominence if prominence > 0 else None,
    )
    if len(peaks) < 2:
        return np.nan, 0.0
    intervals = np.diff(peaks) / fs
    rate = 60.0 / intervals.mean()
    if not (BAND_LOW_BPM <= rate <= BAND_HIGH_BPM):
        return np.nan, 0.0
    cv = intervals.std() / intervals.mean()
    return rate, 1.0 / (1.0 + cv)


def zero_crossing_method(window, fs):
    """Positive-going zero crossings: rate from the mean inter-crossing
    interval, quality from how regular those intervals are."""
    signs = np.sign(window)
    signs[signs == 0] = 1
    crossings = np.flatnonzero((signs[:-1] < 0) & (signs[1:] > 0))
    if len(crossings) < 2:
        return np.nan, 0.0
    intervals = np.diff(crossings) / fs
    rate = 60.0 / intervals.mean()
    if not (BAND_LOW_BPM <= rate <= BAND_HIGH_BPM):
        return np.nan, 0.0
    cv = intervals.std() / intervals.mean()
    return rate, 1.0 / (1.0 + cv)


def spectral_method(window, fs):
    """Zero-padded FFT restricted to the breathing band: rate from the
    dominant frequency, quality from how much of the in-band power sits at
    that one peak (spectral dominance ratio)."""
    n = len(window)
    nfft = max(n, int(FFT_ZERO_PAD_S * fs))
    spectrum = np.fft.rfft(window * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    power = np.abs(spectrum) ** 2
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)
    band_power = power[band]
    total = band_power.sum()
    if total <= 0:
        return np.nan, 0.0
    peak_idx = np.argmax(band_power)
    rate = freqs[band][peak_idx] * 60.0
    dominance = band_power[peak_idx] / total
    return rate, dominance


def fuse_window(estimates):
    """Quality-weighted fusion of the 3 methods' (rate, quality) outputs for
    one window. Returns (fused_rate, fused_quality, n_contributing) - fused
    rate is NaN if the window fails the agreement/quality/quorum checks."""
    valid = [(r, q) for r, q in estimates if not np.isnan(r) and q >= METHOD_QUALITY_FLOOR]
    if len(valid) < MIN_METHODS:
        return np.nan, 0.0, len(valid)

    rates = np.array([r for r, _ in valid])
    quals = np.array([q for _, q in valid])
    spread = rates.max() - rates.min()
    fused_quality = quals.mean()

    if spread > AGREEMENT_MAX_BPM or fused_quality < QUALITY_MIN:
        return np.nan, fused_quality, len(valid)

    fused_rate = np.average(rates, weights=quals)
    return fused_rate, fused_quality, len(valid)


def run_ensemble(resp_section, fs):
    filtered = bandpass_filtfilt(resp_section, fs)
    duration_s = len(filtered) / fs
    window_n = int(round(WINDOW_S * fs))

    t_center, rate_fused, quality_fused, n_methods = [], [], [], []
    start = 0.0
    while start + WINDOW_S <= duration_s + 1e-9:
        i0 = int(round(start * fs))
        window = filtered[i0:i0 + window_n]
        if len(window) < window_n:
            break

        fused, quality, n_valid = fuse_window([
            peak_method(window, fs),
            zero_crossing_method(window, fs),
            spectral_method(window, fs),
        ])
        t_center.append(start + WINDOW_S / 2.0)
        rate_fused.append(fused)
        quality_fused.append(quality)
        n_methods.append(n_valid)
        start += SHIFT_S

    return (
        np.array(t_center), np.array(rate_fused),
        np.array(quality_fused), np.array(n_methods), filtered,
    )


def smooth_for_display(t_center, rate_fused):
    """Bridge rejected (NaN) windows and tame window-to-window jitter for
    display purposes only - the underlying per-window accept/reject record in
    rate_fused is what's trustworthy; this is a cosmetic smoothing pass on
    top of it so the curve reads as one continuous trend rather than
    scattered dots. Linearly interpolates across gaps, then applies a
    Savitzky-Golay filter."""
    valid = ~np.isnan(rate_fused)
    if valid.sum() < 2:
        return rate_fused.copy()

    interp = np.interp(t_center, t_center[valid], rate_fused[valid])

    window = min(15, len(interp) if len(interp) % 2 else len(interp) - 1)
    if window >= 5:
        interp = savgol_filter(interp, window_length=window, polyorder=2)
    return interp


def plot_section(subj, section, t_raw, resp_raw, t_center, rate_fused):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.plot(t_raw, resp_raw, linewidth=0.6, color=SERIES_BLUE)
    ax1.set_ylabel("Respiration (a.u.)")
    ax1.set_title(
        f"{subj} - Ensemble respiration rate estimate ({section})",
        loc="left", fontsize=12, color=INK_PRIMARY,
    )

    valid = ~np.isnan(rate_fused)
    rate_smooth = smooth_for_display(t_center, rate_fused)

    ax2.plot(t_center, rate_smooth, linewidth=1.6, color=SERIES_AMBER,
              solid_capstyle="round", label="smoothed trend")
    ax2.scatter(t_center[valid], rate_fused[valid], s=12, color=SERIES_AMBER,
                edgecolors="#fcfcfb", linewidths=0.5, zorder=3,
                label="confident window estimate")
    ax2.legend(loc="upper right", frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY)
    ax2.set_ylim(BAND_LOW_BPM - 4, BAND_HIGH_BPM + 2)
    ax2.set_ylabel("Respiration rate (breaths/min)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(t_raw[0], t_raw[-1])

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{subj}_resp_rate_{section}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for subj in SUBJECTS:
        print(f"\n== {subj} ==")
        data = load_subject(subj)
        resp = data["signal"]["chest"]["Resp"].ravel()
        for section, code in ACTIVITY_CODE.items():
            t0, t1 = section_window(data, code)
            i0, i1 = int(t0 * FS_CHEST), int(t1 * FS_CHEST)
            resp_section = resp[i0:i1]
            t_raw = np.arange(len(resp_section)) / FS_CHEST

            t_center, rate_fused, quality_fused, n_methods, _ = run_ensemble(
                resp_section, FS_CHEST
            )

            valid = ~np.isnan(rate_fused)
            kept_pct = 100.0 * valid.sum() / len(valid) if len(valid) else 0.0
            print(
                f"{section}: {t1 - t0:.0f}s section, {len(valid)} windows, "
                f"{kept_pct:.0f}% kept, "
                f"mean rate {np.nanmean(rate_fused):.1f} breaths/min "
                f"(sd {np.nanstd(rate_fused):.1f})"
            )
            plot_section(subj, section, t_raw, resp_section, t_center, rate_fused)


if __name__ == "__main__":
    main()
