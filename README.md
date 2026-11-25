# Compensatory Audio–EEG Integration for Speech Target Decoding

Implementation of the proposed compensatory audio–EEG model (`Ours`). The model combines frozen HuBERT audio features and EEG features through reliability-guided adaptive weighting.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-preprocessing.txt
python -m pip install -e .
```

## Data and preprocessing

Patient data are not publicly released because they contain sensitive patient information. Speech-condition labels are `CS` (correct speech), `MS` (misarticulated speech), and `AF` (articulation failure). Input formats are provided in `configs/dataset.example.json` and `examples/trials.example.csv`.

Inputs are paired 3-second EEG epochs (32 channels, 500 Hz) and trial-level mono audio (16 kHz). Audio is cropped to the 3-second speech-task interval using synchronized event triggers and provided as trial-level WAV files.

```bash
python preprocess.py eeg --input private/recording.xdf --labels private/trials.csv --output private/S1-D1
```

## Paper settings

| Setting | Value |
| --- | --- |
| Audio encoder | Frozen HuBERT |
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Maximum epochs | 50 |
| Early stopping and checkpoint selection | Training loss |
| Evaluation | Within-session target-stratified 3-fold cross-validation |

The executable configuration, including the random seed, is in `configs/paper.json`.

## Training and evaluation

Specify the dataset manifest using `configs/dataset.example.json`, then run:

```bash
python train.py --config configs/paper.json --manifest private/dataset.json --protocol within-session --output outputs/within_session
python evaluate.py --predictions outputs/within_session/predictions.csv --output outputs/within_session/evaluation
```

Held-out folds are combined within each recording session. Accuracy, macro-F1, and CS/MS/AF target accuracies are calculated per recording session and reported as equally weighted session means ± sample standard deviations (not fold means or pooled-trial accuracy). Outputs include checkpoints, held-out predictions, and session-level metrics.
