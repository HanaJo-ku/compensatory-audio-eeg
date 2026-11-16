import numpy as np
from sklearn.model_selection import StratifiedKFold


def validate_split(metadata, train, test, protocol):
    train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    if not len(train) or not len(test) or set(train) & set(test):
        raise ValueError("Training and test sets must be nonempty and disjoint")
    if min(train.min(), test.min()) < 0 or max(train.max(), test.max()) >= len(metadata):
        raise ValueError("Split index outside dataset")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError("Duplicate split indices")
    a, b = metadata.iloc[train], metadata.iloc[test]
    if set(a.target_id) != set(range(4)):
        raise ValueError("All four target classes must occur in training")
    if protocol == "cross-session" and (set(a.session) & set(b.session) or set(a.participant) != set(b.participant)):
        raise ValueError("Cross-session split must hold out a complete session within a participant")
    if protocol == "cross-subject" and set(a.participant) & set(b.participant):
        raise ValueError("Cross-subject participant leakage")
    if protocol == "within-session" and (a.session.nunique() != 1 or set(a.session) != set(b.session)):
        raise ValueError("Within-session folds must belong to one recording session")
    return train, test


def make_splits(metadata, config):
    protocol, seed = config["protocol"], config["seed"]
    rows = []
    for participant in metadata.participant.drop_duplicates():
        part = metadata[metadata.participant.eq(participant)]
        for session_fold, session in enumerate(part.session.drop_duplicates(), 1):
            indices = part.index[part.session.eq(session)].to_numpy()
            if protocol == "within-session":
                splitter = StratifiedKFold(config["evaluation"]["n_splits"], shuffle=True, random_state=seed)
                for fold, (a, b) in enumerate(splitter.split(indices, metadata.loc[indices, "target_id"]), 1):
                    rows.append(dict(name=f"{participant}/{session}/fold_{fold}", train=indices[a].tolist(), test=indices[b].tolist(), fold=fold))
            elif protocol == "cross-session":
                train = part.index[~part.session.eq(session)].to_numpy()
                rows.append(dict(name=f"{participant}/heldout_{session}", train=train.tolist(), test=indices.tolist(), fold=session_fold))
        if protocol == "cross-subject":
            train = metadata.index[~metadata.participant.eq(participant)].to_numpy()
            test = part.index.to_numpy()
            source = metadata.loc[train, "participant"].drop_duplicates().tolist()
            if len(source) != 1:
                raise ValueError("The supplied cross-subject protocol requires exactly two participants")
            rows.append(dict(name=f"{source[0]}_to_{participant}", train=train.tolist(), test=test.tolist(), fold=1))
    if not rows:
        raise ValueError("No evaluation splits")
    coverage = np.zeros(len(metadata), dtype=int)
    for row in rows:
        train, test = validate_split(metadata, row["train"], row["test"], protocol)
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each trial must be evaluated exactly once")
    return rows
