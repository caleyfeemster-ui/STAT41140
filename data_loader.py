"""
data_loader.py — dataset utilities for weather sequence data
STAT41140: AI for Weather and Climate

This file handles turning raw weather data into sequences the LSTM can learn from.
Currently uses synthetic data so you can run everything without downloading anything.
Swap generate_synthetic_data() for real Met Éireann / EUMetSat data later.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ── Synthetic data generator ──────────────────────────────────────────────────

def generate_synthetic_data(n_days=1000, seed=42):
    """
    Generate fake daily weather data for Ireland.
    Replace this with real data (e.g. Met Éireann CSV) when you have it.

    Returns a dict with arrays of shape (n_days,) for each variable.

    Covariates (inputs):
        temperature   — daily mean temperature (°C)
        pressure      — atmospheric pressure (hPa)
        humidity      — relative humidity (%)
        wind_speed    — wind speed (km/h)

    Target (what we predict):
        rainfall      — next day's rainfall (mm)
    """
    rng = np.random.default_rng(seed)

    # Simulate seasonal patterns with noise
    t = np.linspace(0, 4 * np.pi, n_days)   # ~2 full years
    temperature  = 10 + 6 * np.sin(t) + rng.normal(0, 2, n_days)
    pressure     = 1013 + 5 * np.cos(t) + rng.normal(0, 3, n_days)
    humidity     = 75 + 10 * np.sin(t + 1) + rng.normal(0, 5, n_days)
    wind_speed   = 20 + 8 * np.abs(np.sin(t + 0.5)) + rng.normal(0, 4, n_days)

    # Rainfall: correlated with humidity and inverse pressure, plus noise
    rainfall = (
        0.3 * humidity
        - 0.1 * pressure
        + 0.05 * wind_speed
        + rng.exponential(2, n_days)   # rain is skewed — mostly dry, occasional heavy
    ).clip(0)   # rain can't be negative

    return {
        "temperature": temperature,
        "pressure":    pressure,
        "humidity":    humidity,
        "wind_speed":  wind_speed,
        "rainfall":    rainfall,
    }


# ── Dataset class ─────────────────────────────────────────────────────────────

class WeatherDataset(Dataset):
    """
    Turns a flat array of daily observations into (sequence → target) pairs.

    For each day i, the input is the past `seq_len` days of covariates,
    and the target is rainfall on day i.

    Args:
        data:     dict from generate_synthetic_data() (or real data)
        seq_len:  how many past days to look at (the LSTM's "memory window")
        split:    'train', 'val', or 'test'
    """

    SPLITS = {"train": (0.0, 0.7), "val": (0.7, 0.85), "test": (0.85, 1.0)}

    def __init__(self, data, seq_len=14, split="train"):
        covariate_keys = ["temperature", "pressure", "humidity", "wind_speed"]

        # Stack covariates → (n_days, n_features)
        X_all = np.stack([data[k] for k in covariate_keys], axis=1).astype(np.float32)
        y_all = data["rainfall"].astype(np.float32)

        # Normalise each feature to zero mean, unit variance
        self.X_mean = X_all.mean(axis=0)
        self.X_std  = X_all.std(axis=0) + 1e-8   # avoid division by zero
        X_norm = (X_all - self.X_mean) / self.X_std

        n = len(y_all)
        lo, hi = self.SPLITS[split]
        start, end = int(lo * n), int(hi * n)

        self.X = X_norm[start:end]
        self.y = y_all[start:end]
        self.seq_len = seq_len

    def __len__(self):
        # Number of valid (sequence, target) pairs
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]          # shape: (seq_len, features)
        target = self.y[idx + self.seq_len]                # scalar
        return torch.tensor(x_seq), torch.tensor([target])


# ── Helper to build all three DataLoaders ────────────────────────────────────

def get_dataloaders(seq_len=14, batch_size=32):
    data = generate_synthetic_data()
    loaders = {}
    for split in ("train", "val", "test"):
        ds = WeatherDataset(data, seq_len=seq_len, split=split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),   # only shuffle training data
        )
        print(f"{split:5s}: {len(ds)} samples")
    return loaders


if __name__ == "__main__":
    loaders = get_dataloaders()
    x_batch, y_batch = next(iter(loaders["train"]))
    print(f"\nBatch input shape:  {x_batch.shape}")   # (32, 14, 4)
    print(f"Batch target shape: {y_batch.shape}")    # (32, 1)
