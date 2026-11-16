import re
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import sha256, write_json

CHANNELS = ["Fp1", "Fz", "F3", "F7", "F9", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "Pz", "P3", "P7", "P9", "O1", "Oz", "O2", "P10", "P8", "P4", "CP2", "CP6", "T8", "C4", "Cz", "FC2", "FC6", "F10", "F8", "F4", "Fp2"]
ALIASES = {"F9": ["F9", "FT9"], "P9": ["P9", "TP9"], "P10": ["P10", "TP10"], "F10": ["F10", "FT10"]}


def preprocess(xdf_path, labels_path, output):
    import mne
    import pyxdf
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty EEG preprocessing output directory")
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_streams = [s for s in streams if s["info"]["type"][0].lower() == "eeg"]
    triggers = [s for s in streams if s["info"]["name"][0] == "trigger_stream"]
    if len(eeg_streams) != 1 or len(triggers) != 1:
        raise ValueError("Expected one EEG stream and one trigger_stream")
    stream, events = eeg_streams[0], triggers[0]
    channels = stream["info"]["desc"][0]["channels"][0]["channel"]
    names = [c["label"][0] for c in channels]
    picks = []
    for name in CHANNELS:
        match = [names.index(x) for x in ALIASES.get(name, [name]) if x in names]
        if not match:
            raise ValueError(f"Missing EEG channel {name}")
        picks.append(match[0])
    rate = float(stream["info"]["nominal_srate"][0])
    if rate != 500:
        raise ValueError("The original protocol requires 500 Hz EEG")
    values = np.asarray(stream["time_series"], dtype=np.float64).T[picks]
    times = np.asarray(stream["time_stamps"])
    if not np.isfinite(values).all() or len(times) != values.shape[1] or not np.all(np.diff(times) >= 0):
        raise ValueError("Invalid EEG samples/timestamps")
    raw = mne.io.RawArray(values, mne.create_info(CHANNELS, rate, "eeg"), verbose=False)
    raw.set_montage("standard_1020", on_missing="raise")
    raw.filter(1.0, 50.0, verbose=False)
    filtered = raw.get_data()
    means, stds = filtered.mean(axis=1), filtered.std(axis=1)
    outliers = (abs(means - means.mean()) > 3 * means.std()) | (abs(stds - stds.mean()) > 3 * stds.std())
    bads = [CHANNELS[i] for i in np.flatnonzero(outliers)]
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose=False)
    event_ids = np.asarray(events["time_series"]).flatten().astype(str)
    event_times = np.asarray(events["time_stamps"])
    epochs, boundaries, targets = [], [], []
    block, counts = None, {f"class_{i}": 0 for i in range(1, 5)}
    for marker, onset in zip(event_ids, event_times):
        block_match = re.match(r"start_(\d)", marker.strip())
        if block_match:
            block = block_match.group(1)
        match = re.match(r"class_([1-4])", marker)
        if block != "1" or not match:
            continue
        target = "class_" + match.group(1)
        if counts[target] >= 50:
            continue
        start = int(np.searchsorted(times, onset, side="left"))
        end = start + 1500
        if end > raw.n_times:
            raise ValueError("Incomplete EEG epoch at a selected marker")
        epochs.append(raw.get_data(start=start, stop=end).astype(np.float32))
        boundaries.append(dict(trial_index=len(epochs) - 1, start_sample=start, end_sample=end, event_time=float(onset), duration_sec=3))
        targets.append(target)
        counts[target] += 1
    table = pd.read_csv(labels_path)
    if len(epochs) != 200 or len(table) != len(epochs) or targets != table.target.tolist():
        raise ValueError("Extracted class order must exactly match 200 annotated trial rows")
    if not np.array_equal(table.trial_index, np.arange(len(table))) or not table.state.isin(["CS", "MS", "AF"]).all():
        raise ValueError("Invalid trial annotation indices/states")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "eeg.npy", np.stack(epochs), allow_pickle=False)
    pd.DataFrame(boundaries).to_csv(output / "epoch_boundaries.csv", index=False)
    write_json(output / "preprocessing.json", dict(sampling_rate=rate, bandpass_hz=[1, 50], channels=CHANNELS,
        bad_channels_interpolated=bads, duration_sec=3, selected_block="start_1", trials_per_target=50,
        input_sha256=sha256(xdf_path), labels_sha256=sha256(labels_path)))

