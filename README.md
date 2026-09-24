# Weather LSTM — STAT41140

Deep learning model using LSTM layers to predict Irish weather from historical observations.

**Assignment:** Pick a variable and use an LLM to help write a deep learning model with convolutional and/or LSTM layers in PyTorch to predict a response variable from covariates.

## What this predicts

Currently: **next-day rainfall (mm)** from four covariates:
- Temperature (°C)
- Atmospheric pressure (hPa)
- Relative humidity (%)
- Wind speed (km/h)

Uses synthetic data for now — swap `generate_synthetic_data()` in `data_loader.py` for real Met Éireann or EUMetSat data.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Check the model architecture
python model.py

# Check the data pipeline
python data_loader.py

# Train the model
python train.py
```

## Files

| File | Purpose |
|------|---------|
| `model.py` | LSTM architecture |
| `data_loader.py` | Data generation and batching |
| `train.py` | Training loop |
| `requirements.txt` | Python dependencies |

## Next steps

- [ ] Replace synthetic data with real Met Éireann data
- [ ] Try adding a CNN layer before the LSTM (for spatial features if using gridded data)
- [ ] Experiment with `SEQ_LEN`, `HIDDEN_SIZE`, `NUM_LAYERS` in `train.py`
- [ ] Plot predictions vs actuals
