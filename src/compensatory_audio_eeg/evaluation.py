import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .utils import write_json


def classification_metrics(predictions):
    if predictions.empty or predictions.duplicated(["participant", "session", "trial_index"]).any():
        raise ValueError("Expected unique held-out trial predictions")
    if not predictions.target.isin(range(4)).all() or not predictions.prediction.isin(range(4)).all():
        raise ValueError("Invalid target or prediction IDs")
    sessions = []
    for (participant, session), frame in predictions.groupby(["participant", "session"], sort=True):
        sessions.append(dict(participant=str(participant), session=str(session), n_trials=len(frame),
                             accuracy=100 * accuracy_score(frame.target, frame.prediction),
                             macro_f1=100 * f1_score(frame.target, frame.prediction, labels=range(4), average="macro", zero_division=0)))
    if "state" in predictions:
        if not predictions.state.isin(["CS", "MS", "AF"]).all():
            raise ValueError("Expected CS, MS or AF speech-condition labels")
        for row in sessions:
            frame = predictions[predictions.participant.astype(str).eq(row["participant"]) & predictions.session.astype(str).eq(row["session"])]
            for state in ("CS", "MS", "AF"):
                subset = frame[frame.state.eq(state)]
                row[state] = float(100 * subset.target.eq(subset.prediction).mean()) if len(subset) else None
    values = pd.DataFrame(sessions)
    result = dict(aggregation="recording_session", n_trials=len(predictions), n_sessions=len(values), sessions=sessions)
    for metric in ["accuracy", "macro_f1"] + [state for state in ("CS", "MS", "AF") if state in values]:
        result[metric] = float(values[metric].mean()) if values[metric].notna().any() else None
        result[metric + "_sd"] = float(values[metric].std(ddof=1)) if values[metric].notna().sum() > 1 else None
    return result


def format_metrics(metrics):
    parts = []
    for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")):
        sd = metrics[key + "_sd"]
        spread = f" ± {sd:.2f}" if sd is not None else ""
        parts.append(f"{label} {metrics[key]:.2f}{spread}%")
    return "Session mean: " + "  ".join(parts)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = classification_metrics(pd.read_csv(args.predictions))
    write_json(args.output / "metrics.json", metrics)
    print(format_metrics(metrics))
