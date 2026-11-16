import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compensatory_audio_eeg.data import normalize_eeg, TrialDataset
from compensatory_audio_eeg.evaluation import classification_metrics
from compensatory_audio_eeg.models import Ours
from compensatory_audio_eeg.splits import make_splits, validate_split
from compensatory_audio_eeg.training import make_losses, run_epoch, train_fold
from compensatory_audio_eeg.utils import restore_rng, rng_state, seed_everything


class TinyAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.proj = nn.Linear(1, 8)

    def forward(self, input_values, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.proj(input_values.unsqueeze(-1)))


class TinyEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_dim = 8
        self.proj = nn.Linear(32, 8)

    def forward(self, x):
        return self.proj(x.mean(-1))


class TinyExtractor:
    def __call__(self, waves, **kwargs):
        n = max(map(len, waves))
        inputs = torch.tensor(np.stack([np.pad(x, (0, n - len(x))) for x in waves]), dtype=torch.float32)
        return SimpleNamespace(input_values=inputs, attention_mask=torch.ones_like(inputs, dtype=torch.long))


def tiny_model(config, device):
    return Ours(TinyAudio(), TinyEEG()).to(device), TinyExtractor()


def configuration():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/paper.json").read_text())


def metadata():
    rows = []
    for participant in ("S1", "S2"):
        for day in (1, 2):
            for i in range(12):
                rows.append(dict(participant=participant, session=f"{participant}-D{day}", trial_index=i,
                    state=["CS", "MS", "AF"][i % 3], target=f"class_{i % 4 + 1}", state_id=i % 3, target_id=i % 4))
    return pd.DataFrame(rows)


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_protocols_have_disjoint_and_complete_held_out_trials(self):
        data = metadata()
        for protocol in ("within-session", "cross-session", "cross-subject"):
            cfg = configuration()
            cfg["protocol"] = protocol
            splits = make_splits(data, cfg)
            evaluated = [i for split in splits for i in split["test"]]
            self.assertEqual(sorted(evaluated), list(range(len(data))))
            if protocol == "cross-session":
                self.assertEqual([s["fold"] for s in splits], [1, 2, 1, 2])
            if protocol == "cross-subject":
                self.assertEqual([s["fold"] for s in splits], [1, 1])
            for split in splits:
                self.assertFalse(set(split["train"]) & set(split["test"]))
                if protocol == "cross-subject":
                    self.assertFalse(set(data.iloc[split["train"]].participant) & set(data.iloc[split["test"]].participant))
                if protocol == "cross-session":
                    self.assertFalse(set(data.iloc[split["train"]].session) & set(data.iloc[split["test"]].session))

    def test_target_subject_leakage_is_rejected(self):
        data = metadata()
        with self.assertRaises(ValueError):
            validate_split(data, list(range(36)), list(range(36, 48)), "cross-subject")

    def test_loss_weights_ignore_held_out_labels(self):
        data = metadata()
        changed = data.copy()
        changed.loc[24:, ["state_id", "target_id"]] = 0
        a, _ = make_losses(data, np.arange(24), configuration(), torch.device("cpu"))
        b, _ = make_losses(changed, np.arange(24), configuration(), torch.device("cpu"))
        self.assertTrue(torch.equal(a.state_weights, b.state_weights))
        self.assertTrue(torch.equal(a.target_weights, b.target_weights))

    def test_gate_endpoints_select_the_corresponding_branch(self):
        seed_everything(56)
        model, _ = tiny_model(configuration(), torch.device("cpu"))
        model.eval()
        eeg = torch.randn(2, 32, 64)
        inputs = dict(input_values=torch.randn(2, 64))
        with torch.no_grad():
            model.gate_mlp[-1].weight.zero_()
            for bias, branch in ((-100.0, "audio_logits"), (100.0, "eeg_logits")):
                model.gate_mlp[-1].bias.fill_(bias)
                out = model(eeg, inputs)
                self.assertTrue(torch.allclose(out["final_logits"].softmax(1), out[branch].softmax(1), atol=1e-6))
        self.assertFalse(any(p.requires_grad for p in model.audio_encoder.parameters()))

    def test_basic_metrics_and_duplicate_prediction_check(self):
        frame = pd.DataFrame(dict(participant=["S1"] * 4, session=["S1-D1"] * 4,
                                  trial_index=range(4), target=[0, 1, 2, 3], prediction=[0, 1, 3, 2]))
        result = classification_metrics(frame)
        self.assertEqual(result["accuracy"], 50.0)
        self.assertEqual(result["macro_f1"], 50.0)
        with self.assertRaises(ValueError):
            classification_metrics(pd.concat([frame, frame.iloc[:1]]))

    def test_metrics_weight_recording_sessions_equally(self):
        frame = pd.DataFrame(dict(participant=["S1"] * 4, session=["D1", "D2", "D2", "D2"],
                                  trial_index=[0, 0, 1, 2], target=[0, 0, 1, 2], prediction=[0, 1, 2, 3]))
        result = classification_metrics(frame)
        self.assertEqual(result["accuracy"], 50.0)
        self.assertAlmostEqual(result["accuracy_sd"], 70.71067811865476)
        self.assertEqual(result["n_sessions"], 2)

    def test_dataset_requires_three_second_inputs(self):
        data = metadata()
        eeg = [np.zeros((32, 1500), dtype=np.float32)]
        audio = [np.zeros(48000, dtype=np.float32)]
        sample = TrialDataset(eeg, audio, data, [0])[0]
        self.assertEqual(tuple(sample["eeg"].shape), (32, 1500))
        with self.assertRaises(ValueError):
            TrialDataset([np.zeros((32, 1000), dtype=np.float32)], audio, data, [0])[0]

    def test_rng_round_trip_and_eeg_normalization(self):
        seed_everything(56)
        saved = rng_state()
        a = torch.rand(4)
        b = np.random.rand(4)
        restore_rng(saved)
        self.assertTrue(torch.equal(a, torch.rand(4)))
        self.assertTrue(np.array_equal(b, np.random.rand(4)))
        values = np.arange(48, dtype=np.float32).reshape(2, 3, 8)
        normal = normalize_eeg(values)
        self.assertTrue(np.allclose(normal.mean(-1), 0, atol=1e-6))
        self.assertTrue(np.allclose(normal.std(-1), 1, atol=1e-5))

    def test_offline_training_evaluates_only_after_selection(self):
        cfg = configuration()
        cfg["training"].update(max_epochs=2, min_epochs=2, patience=2, amp=False)
        cfg["protocol"] = "within-session"
        data = metadata()
        split = make_splits(data, cfg)[0]
        random = np.random.default_rng(56)
        eeg = [random.normal(size=(32, 1500)).astype(np.float32) for _ in range(len(data))]
        audio = [random.normal(size=48000).astype(np.float32) for _ in range(len(data))]
        from rich.console import Console
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "fold"
            backbone = Path(temp) / "frozen_audio.pt"
            with patch("compensatory_audio_eeg.training.build_ours", side_effect=tiny_model), patch("compensatory_audio_eeg.training.run_epoch", wraps=run_epoch) as calls:
                output = train_fold(eeg, audio, data, split, cfg, torch.device("cpu"), folder, backbone, Console(quiet=True))
            evaluation_calls = [c for c in calls.call_args_list if len(c.args) <= 6]
            self.assertEqual(len(evaluation_calls), 1)
            self.assertEqual(len(output), len(split["test"]))
            self.assertTrue(np.array_equal(output.trial_index, data.iloc[split["test"]].trial_index))
            self.assertTrue((folder / "complete.json").exists())
            state = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
            self.assertFalse(any(k.startswith("audio_encoder.") for k in state["model_state_dict"]))
            self.assertTrue(backbone.exists())


if __name__ == "__main__":
    unittest.main()
