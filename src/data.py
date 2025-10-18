import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = {0: "sano", 1: "malato"}

def _clip_features(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    # [battito, saturazione, pressione, temperatura]
    X[:, 0] = np.clip(X[:, 0], 40, 180)     # bpm
    X[:, 1] = np.clip(X[:, 1], 70, 100)     # SpO2 %
    X[:, 2] = np.clip(X[:, 2], 70, 200)     # sistolica mmHg
    X[:, 3] = np.clip(X[:, 3], 34.0, 41.0)  # °C
    return X

def _sample_block(rng: np.random.Generator, means_stds: dict, n: int) -> np.ndarray:
    return np.column_stack([
        rng.normal(*means_stds["battito"], size=n),
        rng.normal(*means_stds["saturazione"], size=n),
        rng.normal(*means_stds["pressione"], size=n),
        rng.normal(*means_stds["temperatura"], size=n),
    ])

def _make_medical_dataset(n_samples=5000, ratio_sani=0.6, seed=42, add_noise_prob=0.02):
    rng = np.random.default_rng(seed)
    n_sani = int(n_samples * ratio_sani)
    n_malati = n_samples - n_sani

    sano = {
        "battito": (75, 8),
        "saturazione": (97, 1.2),
        "pressione": (120, 10),
        "temperatura": (36.7, 0.25),
    }
    malato = {
        "battito": (100, 15),
        "saturazione": (91, 3.0),
        "pressione": (100, 15),
        "temperatura": (38.1, 0.6),
    }

    X_sani   = _sample_block(rng, sano,   n_sani)
    X_malati = _sample_block(rng, malato, n_malati)

    # semplice correlazione: saturazione più bassa -> battito più alto (malati)
    tachy_boost = np.maximum(0, 97 - X_malati[:, 1]) * rng.normal(1.2, 0.2, size=n_malati)
    X_malati[:, 0] += tachy_boost

    X_sani   = _clip_features(X_sani)
    X_malati = _clip_features(X_malati)

    X = np.vstack([X_sani, X_malati]).astype(np.float32)
    y = np.concatenate([np.zeros(n_sani, dtype=np.int64), np.ones(n_malati, dtype=np.int64)])

    # mescola
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # rumore raro
    if add_noise_prob > 0:
        mask = rng.random(len(X)) < add_noise_prob
        X[mask, 0] += rng.normal(0, 25, mask.sum())  # battito
        X[mask, 1] += rng.normal(0, 4,  mask.sum())  # SpO2
        X = _clip_features(X)

    return X, y

def make_loaders(batch_size=64, test_size=0.2, ratio_sani=0.6, seed=42, return_scaler=False):
    X, y = _make_medical_dataset(n_samples=5000, ratio_sani=ratio_sani, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)

    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False)

    if return_scaler:
        return train_loader, test_loader, scaler
    return train_loader, test_loader
