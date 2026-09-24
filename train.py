"""
train.py — training loop for the weather LSTM
STAT41140: AI for Weather and Climate

Run this file to train the model:
    python train.py

It will print loss at each epoch and save the best model to best_model.pt
"""

import torch
import torch.nn as nn
from model import WeatherLSTM
from data_loader import get_dataloaders

# ── Hyperparameters ───────────────────────────────────────────────────────────
# These are the knobs you can turn to improve the model.

SEQ_LEN     = 14        # how many past days the LSTM looks at
BATCH_SIZE  = 32        # how many examples per gradient update
HIDDEN_SIZE = 64        # width of the LSTM's hidden state
NUM_LAYERS  = 2         # number of stacked LSTM layers
DROPOUT     = 0.2       # regularisation — helps prevent overfitting
LEARNING_RATE = 0.001   # step size for gradient descent
EPOCHS      = 30        # number of full passes through the training data


def train():
    # ── Data ─────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WeatherLSTM(
        input_size=4,           # 4 covariates: temp, pressure, humidity, wind
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,          # predicting 1 value: rainfall
        dropout=DROPOUT,
    )

    # ── Loss and optimiser ────────────────────────────────────────────────────
    criterion = nn.MSELoss()                          # mean squared error
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(f"\nTraining for {EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10}")
    print("-" * 36)

    for epoch in range(1, EPOCHS + 1):

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in loaders["train"]:
            optimiser.zero_grad()         # clear old gradients
            predictions = model(x_batch)  # forward pass
            loss = criterion(predictions, y_batch)
            loss.backward()               # backward pass — compute gradients
            optimiser.step()              # update weights
            train_loss += loss.item()

        train_loss /= len(loaders["train"])

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():   # no gradients needed for validation
            for x_batch, y_batch in loaders["val"]:
                predictions = model(x_batch)
                val_loss += criterion(predictions, y_batch).item()

        val_loss /= len(loaders["val"])

        # ── Logging ──────────────────────────────────────────────────────────
        marker = " ← best" if val_loss < best_val_loss else ""
        print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>10.4f}{marker}")

        # Save the model whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nDone. Best validation loss: {best_val_loss:.4f}")
    print("Model saved to best_model.pt")


if __name__ == "__main__":
    train()
