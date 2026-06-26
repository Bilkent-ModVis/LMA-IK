# Model checkpoints

The trained model weights are included in this directory and tracked in git:

- `interpolator.pth` — Interpolator CVAE (3.30 M parameters)
- `synthesizer.pth`  — Synthesizer LSTM (5.62 M parameters, matching Table 3)

`reproduce.py` loads both checkpoints from here.
