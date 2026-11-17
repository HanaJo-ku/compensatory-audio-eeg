# Evaluation Splits

These files contain the anonymized evaluation splits used in the reported experiments.

- `cv3_test_assignments.csv`: Patient, session, trial ID, and test-fold assignment for within-session three-fold cross-validation. Trial IDs are session-local. For each test fold, all remaining trials in the same session form the training set.
- `cv3_fold_counts.csv`: Test-trial counts for each session and fold.
- `loso_held_out_sessions.csv`: Held-out sessions for within-patient leave-one-session-out evaluation. All other sessions from the same patient form the training set.
- `loso_session_composition.csv`: Patient–session membership and trial counts.

Early stopping and checkpoint selection were based on training loss; held-out evaluation data were not used for checkpoint selection.

These files contain split metadata only. Patient audio, EEG, and speech-target/state annotations are not publicly released.
