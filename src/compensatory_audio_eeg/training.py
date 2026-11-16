import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel
from rich.console import Console

from .data import TrialDataset, collator, load_data
from .evaluation import classification_metrics, format_metrics
from .losses import StateBalancedLoss
from .models import Ours, EEGModule
from .splits import make_splits
from .utils import environment, generator, read_json, restore_rng, rng_state, save_torch, seed_everything, sha256, write_json


def build_ours(config, device):
    spec = config["model"]
    extractor = AutoFeatureExtractor.from_pretrained(spec["audio_model"], revision=spec["audio_revision"])
    backbone = AutoModel.from_pretrained(spec["audio_model"], revision=spec["audio_revision"], use_safetensors=False, weights_only=True)
    eeg_encoder = EEGModule(n_channels=spec["n_channels"], d_model=spec["eeg_d_model"], dropout=spec["eeg_dropout"])
    model = Ours(backbone, eeg_encoder, freeze_audio_encoder=True,
                                     fusion_dim=spec["fusion_dim"], n_target=spec["n_targets"])
    model.gate_mlp[3].p, model.gate_mlp[6].p = spec["gate_dropout"]
    model.audio_head[3].p = spec["audio_head_dropout"]
    model.eeg_head[3].p = spec["eeg_head_dropout"]
    return model.to(device), extractor


def make_losses(metadata, train, config, device):
    settings = config["training"]
    target = metadata.iloc[train].target_id.to_numpy()
    state = metadata.iloc[train].state_id.to_numpy()
    criterion = StateBalancedLoss(state, target, 4, settings["condition_weight_power"], settings["label_smoothing"]).to(device)
    counts = np.bincount(target, minlength=4).astype(np.float32)
    counts[counts == 0] = 1
    weights = torch.tensor(counts.sum() / (4 * counts), device=device)
    auxiliary = nn.CrossEntropyLoss(weight=weights, label_smoothing=settings["label_smoothing"])
    return criterion, auxiliary


def run_epoch(model, loader, config, device, criterion, auxiliary, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    predictions = []
    settings = config["training"]
    amp = settings["amp"] and device.type == "cuda"
    for batch in loader:
        eeg = batch["eeg"].to(device)
        inputs = {k: v.to(device) for k, v in batch["encoder_inputs"].items()}
        targets, states = batch["target"].to(device), batch["state"].to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device.type, enabled=amp):
                outputs = model(eeg, inputs)
                loss = criterion(outputs["final_logits"], targets, states)
                loss = loss + settings["audio_aux_weight"] * auxiliary(outputs["audio_logits"], targets)
                loss = loss + settings["eeg_aux_weight"] * auxiliary(outputs["eeg_logits"], targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total += loss.item() * len(targets)
        if not training:
            audio = outputs["audio_logits"].softmax(1).float().cpu().numpy()
            eeg_prob = outputs["eeg_logits"].softmax(1).float().cpu().numpy()
            final = outputs["final_logits"].softmax(1).float().cpu().numpy()
            gates = outputs["gate_prob"].float().cpu().numpy()
            for i in range(len(targets)):
                row = dict(target=int(batch["target"][i]), state=("CS", "MS", "AF")[int(batch["state"][i])],
                           prediction=int(final[i].argmax()), audio_prediction=int(audio[i].argmax()),
                           eeg_prediction=int(eeg_prob[i].argmax()), gate_weight=float(gates[i]))
                for name, probabilities in (("final", final[i]), ("audio", audio[i]), ("eeg", eeg_prob[i])):
                    for j, probability in enumerate(probabilities):
                        row[f"{name}_prob_{j}"] = float(probability)
                predictions.append(row)
    return total / len(loader.dataset), pd.DataFrame(predictions)


def train_fold(eeg, audio, metadata, split, config, device, output, backbone_path, console):
    output.mkdir(parents=True, exist_ok=True)
    model, extractor = build_ours(config, device)
    if not backbone_path.exists():
        save_torch(backbone_path, model.audio_encoder.state_dict())
    else:
        model.audio_encoder.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
    settings = config["training"]
    collate = collator(extractor, config["data"]["audio_sampling_rate"])
    train_loader = DataLoader(TrialDataset(eeg, audio, metadata, split["train"]),
        batch_size=settings["batch_size"], shuffle=True, num_workers=0, collate_fn=collate,
        generator=generator(config["seed"] + split["fold"]))
    test_loader = DataLoader(TrialDataset(eeg, audio, metadata, split["test"]),
        batch_size=settings["batch_size"], shuffle=False, num_workers=0, collate_fn=collate,
        generator=generator(config["seed"] + 20000 + split["fold"]))
    criterion, auxiliary = make_losses(metadata, split["train"], config, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=settings["scheduler_factor"], patience=settings["scheduler_patience"])
    scaler = torch.amp.GradScaler(device.type, enabled=settings["amp"] and device.type == "cuda")
    best, best_epoch, waiting, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, settings["max_epochs"] + 1):
        loss, _ = run_epoch(model, train_loader, config, device, criterion, auxiliary, optimizer, scaler)
        if not np.isfinite(loss):
            raise ValueError("Non-finite training loss")
        scheduler.step(-loss)
        if loss < best:
            best, best_epoch, waiting = loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith("audio_encoder.")}
        else:
            waiting += 1
        history.append(dict(epoch=epoch, training_loss=loss, learning_rate=optimizer.param_groups[0]["lr"]))
        if epoch == 1 or epoch % settings["log_every"] == 0:
            console.print(f"  {epoch:3d}  train loss {loss:.4f}  best {best:.4f}")
        if epoch >= settings["min_epochs"] and waiting >= settings["patience"]:
            break
    missing, unexpected = model.load_state_dict(best_state, strict=False)
    if unexpected or any(not k.startswith("audio_encoder.") for k in missing):
        raise ValueError("Incomplete best checkpoint")
    _, predictions = run_epoch(model, test_loader, config, device, criterion, auxiliary)
    held = metadata.iloc[split["test"]].reset_index(drop=True)
    predictions = pd.concat([held[["participant", "session", "trial_index"]], predictions], axis=1)
    predictions["fold"] = split["fold"]
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    write_json(output / "split.json", split)
    save_torch(output / "model.pt", dict(model_state_dict=best_state, backbone="frozen_audio.pt", best_epoch=best_epoch,
        best_training_loss=best, config=config, split=split, rng=rng_state()))
    write_json(output / "complete.json", dict(best_epoch=best_epoch, best_training_loss=best,
        prediction_sha256=sha256(output / "predictions.csv")))
    console.print(f"  [#d78eb5]Saved epoch {best_epoch}; held-out accuracy {100 * predictions.target.eq(predictions.prediction).mean():.2f}%[/]")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["within-session", "cross-session", "cross-subject"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.protocol:
        config["protocol"] = args.protocol
    if not config["model"]["freeze_audio_encoder"] or config["evaluation"]["selection"] != "training_loss":
        parser.error("This release implements the frozen-HuBERT, training-loss-selection protocol")
    if config["data"]["crop_sec"] != 0 or not config["data"]["normalize_eeg_per_trial"]:
        parser.error("This release reproduces the original full-epoch per-trial-normalized model")
    if config["data"]["epoch_duration_sec"] != 3 or config["data"]["eeg_sampling_rate"] != 500 or config["data"]["audio_sampling_rate"] != 16000:
        parser.error("Inputs must be 3-second EEG at 500 Hz and audio at 16000 Hz")
    eeg, audio, metadata, fingerprints = load_data(args.manifest, config)
    splits = make_splits(metadata, config)
    if args.dry_run:
        for split in splits:
            print(split["name"], len(split["train"]), len(split["test"]))
        return
    args.output = args.output.resolve()
    exists = args.output.exists() and any(args.output.iterdir())
    if exists and not args.resume:
        parser.error("Output exists; use --resume or a new output directory")
    if exists:
        if read_json(args.output / "config.json") != config or read_json(args.output / "input_fingerprints.json") != fingerprints:
            parser.error("Saved settings or input contents differ; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "environment.json", environment())
    write_json(args.output / "input_fingerprints.json", fingerprints)
    write_json(args.output / "splits.json", splits)
    console = Console(highlight=False)
    device = torch.device(args.device)
    seed_everything(config["seed"])
    all_predictions = []
    for split in splits:
        folder = args.output / "folds" / split["name"]
        if config["protocol"] == "cross-subject":
            seed_everything(config["seed"])
        if args.resume and (folder / "complete.json").exists():
            if sha256(folder / "predictions.csv") != read_json(folder / "complete.json")["prediction_sha256"]:
                raise ValueError("Saved predictions changed since completion")
            state = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
            restore_rng(state["rng"])
            predictions = pd.read_csv(folder / "predictions.csv")
            console.print(f"Skipped complete: {split['name']}", style="dim")
        else:
            console.print(f"[bold #d78eb5]{split['name']}[/]  train n={len(split['train'])}  test n={len(split['test'])}")
            predictions = train_fold(eeg, audio, metadata, split, config, device, folder, args.output / "frozen_audio.pt", console)
        all_predictions.append(predictions)
        combined = pd.concat(all_predictions, ignore_index=True)
        combined.to_csv(args.output / "predictions.partial.csv", index=False)
    oof = pd.concat(all_predictions, ignore_index=True)
    oof.to_csv(args.output / "predictions.csv", index=False)
    metrics = classification_metrics(oof)
    write_json(args.output / "metrics.json", metrics)
    console.print(format_metrics(metrics))
    write_json(args.output / "complete.json", dict(n_trials=len(oof), prediction_sha256=sha256(args.output / "predictions.csv")))
