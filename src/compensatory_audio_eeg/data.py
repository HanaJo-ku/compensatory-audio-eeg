from pathlib import Path
import re

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .utils import read_json, sha256

STATES = {"CS": 0, "MS": 1, "AF": 2}
TARGETS = {f"class_{i}": i - 1 for i in range(1, 5)}


def normalize_eeg(eeg):
    return (eeg - eeg.mean(axis=-1, keepdims=True)) / (eeg.std(axis=-1, keepdims=True) + 1e-6)


def resolve(base, name):
    path = Path(name)
    return path if path.is_absolute() else base / path


def load_data(manifest_path, config):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    eeg, audio, records, fingerprints = [], [], [], []
    sessions = manifest["sessions"]
    if not sessions or len({s["session"] for s in sessions}) != len(sessions):
        raise ValueError("Session identifiers must be nonempty and unique")
    for session in sessions:
        for key in ("participant", "session"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session[key]) or session[key] in (".", ".."): 
                raise ValueError("Participant/session IDs must be simple anonymous identifiers")
        eeg_path = resolve(manifest_path.parent, session["eeg"])
        table_path = resolve(manifest_path.parent, session["trials"])
        values = np.load(eeg_path, allow_pickle=False)
        table = pd.read_csv(table_path)
        required = {"trial_index", "target", "state", "audio"}
        if required - set(table.columns):
            raise ValueError(f"Trial metadata requires {sorted(required)}")
        if session.get("duration_sec", 3) != 3:
            raise ValueError("All EEG and audio trials must use 3-second inputs")
        expected = 3 * config["data"]["eeg_sampling_rate"]
        if values.ndim != 3 or values.shape[1:] != (config["model"]["n_channels"], expected):
            raise ValueError(f"EEG must have shape (trials, channels, duration_samples): {session['session']}")
        if len(table) != len(values) or table.trial_index.duplicated().any() or not np.array_equal(table.trial_index, np.arange(len(values))):
            raise ValueError("EEG rows and consecutive trial indices must match exactly")
        if not np.isfinite(values).all() or not table.state.isin(STATES).all() or not table.target.isin(TARGETS).all():
            raise ValueError("Non-finite EEG or invalid trial labels")
        values = normalize_eeg(values.astype(np.float32))
        eeg.extend(list(values))
        fingerprints.extend([dict(kind="eeg", session=session["session"], sha256=sha256(eeg_path)),
                             dict(kind="trials", session=session["session"], sha256=sha256(table_path))])
        for row in table.itertuples(index=False):
            audio_path = resolve(table_path.parent, row.audio)
            wave, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            target_sr = config["data"]["audio_sampling_rate"]
            if sr != target_sr:
                divisor = np.gcd(sr, target_sr)
                wave = resample_poly(wave, target_sr // divisor, sr // divisor).astype(np.float32)
            if len(wave) != 3 * target_sr:
                raise ValueError("Expected a 3-second trial WAV cropped using event triggers")
            wave = wave.astype(np.float32)
            if not len(wave) or not np.isfinite(wave).all():
                raise ValueError("Invalid audio waveform")
            audio.append(wave)
            fingerprints.append(dict(kind="audio", session=session["session"], trial=int(row.trial_index), sha256=sha256(audio_path)))
            records.append(dict(participant=session["participant"], session=session["session"],
                                trial_index=int(row.trial_index), state=row.state, target=row.target,
                                state_id=STATES[row.state], target_id=TARGETS[row.target]))
    return eeg, audio, pd.DataFrame(records), fingerprints


class TrialDataset(Dataset):
    def __init__(self, eeg, audio, metadata, indices):
        self.eeg, self.audio, self.meta = eeg, audio, metadata
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        eeg = self.eeg[i]
        if eeg.shape != (32, 1500) or self.audio[i].shape != (48000,):
            raise ValueError("Expected 3-second EEG (32, 1500) and audio (48000,)")
        return dict(eeg=torch.tensor(eeg, dtype=torch.float32), audio=self.audio[i],
                    state=int(self.meta.iloc[i].state_id), target=int(self.meta.iloc[i].target_id))


def collator(extractor, sampling_rate):
    def collate(batch):
        inputs = extractor([row["audio"] for row in batch], sampling_rate=sampling_rate,
                           padding=True, return_attention_mask=True, return_tensors="pt")
        encoder_inputs = {"input_values": inputs.input_values}
        if hasattr(inputs, "attention_mask"):
            encoder_inputs["attention_mask"] = inputs.attention_mask
        return dict(eeg=torch.stack([row["eeg"] for row in batch]), encoder_inputs=encoder_inputs,
                    state=torch.tensor([row["state"] for row in batch]), target=torch.tensor([row["target"] for row in batch]))
    return collate
