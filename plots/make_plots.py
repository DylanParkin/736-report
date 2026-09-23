"""
Generate report plots from the PPG-DaLiA field-study recordings for
subjects S7 and S8 (S7.pkl / S8.pkl).

For each subject, two activity sections (sitting / walking), each with
three plots sharing the exact same absolute time window, so HR, PPG and
respiration stay synchronised for that section:
  1. Reference heart rate vs time       (data['label'],                  ~0.5 Hz)
  2. Wrist PPG (BVP) signal vs time     (data['signal']['wrist']['BVP'],  64 Hz)
  3. Respiration vs time, from RespiBAN (data['signal']['chest']['Resp'], 700 Hz)

Sections are taken from data['activity'] (4 Hz, synchronised with the
RespiBAN start time): activity code 1 = BASELINE (sitting still), code 7 =
WALKING, per the dataset readme (Section II) and confirmed against each
subject's SX_activity.csv.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "S7_S8_data", "PPG_FieldStudy")
OUT_DIR = BASE_DIR

SUBJECTS = ["S7", "S8"]

FS_CHEST = 700.0     # RespiBAN chest unit (ECG/EMG/EDA/Temp/Resp/ACC)
FS_BVP = 64.0        # Empatica E4 wrist PPG
FS_ACTIVITY = 4.0    # activity-code signal, synced to RespiBAN start time

ACTIVITY_CODE = {"sitting": 1, "walking": 7}  # BASELINE, WALKING (readme Sec. II)

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"

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
    """Absolute [start_s, end_s) of the (single) activity segment with
    the given code, in RespiBAN/chest time - the common time base every
    signal in SX.pkl is synchronised to."""
    segments = activity_segments(data["activity"].ravel(), FS_ACTIVITY)
    matches = [(s, e) for c, s, e in segments if c == code]
    return max(matches, key=lambda se: se[1] - se[0])  # longest run of that code


def line_plot(t, y, title, ylabel, out_path, color=SERIES_BLUE, linewidth=0.7):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, y, linewidth=linewidth, color=color)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_hr(subj, data, section, t0, t1):
    label = data["label"].ravel()
    # Per readme III.3: each label is the mean HR over an 8 s window,
    # shifted 2 s between windows -> label[i] covers [i*2, i*2+8) s.
    # Use each window's centre time, in the same absolute time base as
    # the activity segment.
    window_len_s, window_shift_s = 8.0, 2.0
    t_abs = np.arange(len(label)) * window_shift_s + window_len_s / 2
    mask = (t_abs >= t0) & (t_abs < t1)
    line_plot(
        t_abs[mask] - t0, label[mask],
        title=f"{subj} - Reference heart rate vs time ({section})",
        ylabel="Heart rate (bpm)",
        out_path=os.path.join(OUT_DIR, f"{subj}_heart_rate_{section}.png"),
    )


def plot_ppg(subj, data, section, t0, t1):
    bvp = data["signal"]["wrist"]["BVP"].ravel()
    t_abs = np.arange(len(bvp)) / FS_BVP
    mask = (t_abs >= t0) & (t_abs < t1)
    line_plot(
        t_abs[mask] - t0, bvp[mask],
        title=f"{subj} - Wrist PPG (BVP) signal vs time ({section})",
        ylabel="BVP (a.u.)",
        out_path=os.path.join(OUT_DIR, f"{subj}_ppg_signal_{section}.png"),
        linewidth=0.6,
    )


def plot_resp(subj, data, section, t0, t1):
    resp = data["signal"]["chest"]["Resp"].ravel()
    t_abs = np.arange(len(resp)) / FS_CHEST
    mask = (t_abs >= t0) & (t_abs < t1)
    line_plot(
        t_abs[mask] - t0, resp[mask],
        title=f"{subj} - Respiration vs time, RespiBAN ({section})",
        ylabel="Respiration (a.u.)",
        out_path=os.path.join(OUT_DIR, f"{subj}_respiration_{section}.png"),
        linewidth=0.6,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for subj in SUBJECTS:
        print(f"\n== {subj} ==")
        data = load_subject(subj)
        for section, code in ACTIVITY_CODE.items():
            t0, t1 = section_window(data, code)
            print(f"{section}: {t0:.1f}s - {t1:.1f}s ({t1 - t0:.1f}s)")
            plot_hr(subj, data, section, t0, t1)
            plot_ppg(subj, data, section, t0, t1)
            plot_resp(subj, data, section, t0, t1)


if __name__ == "__main__":
    main()
