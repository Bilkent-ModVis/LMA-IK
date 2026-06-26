#!/usr/bin/env python3
"""Reproduce the LMA descriptor-effect result of the paper (Figs 4-8).

Runs with NO parameters:

    python reproduce.py

It loads the released ``interpolator.pth`` / ``synthesizer.pth`` checkpoints,
runs the full two-stage data-driven IK pipeline (Interpolator -> Synthesizer ->
forward kinematics) on a set of base motions from the LMA Effort dataset, and
generates full-body motion for LOW vs HIGH values of each LMA style descriptor
(V, H, P).  For every descriptor it then *measures* the realized descriptor on
the generated motion and reports the mean over the base motions, demonstrating
that conditioning on a higher descriptor value yields a higher measured value --
the effect illustrated qualitatively in Figs 5(b), 6 and 7 of the paper.

Outputs (written to ``results/``):
  * ``descriptor_effect.txt``  -- human-readable summary table
  * ``descriptor_effect.json`` -- the same data in machine-readable form
  * ``motion_<D>_low.csv`` / ``motion_<D>_high.csv`` -- the generated full-body
    joint world positions (frame, joint, x, y, z) for one example base motion,
    i.e. the data underlying the low/high pose comparison figures.

The script auto-extracts the dataset zip and builds the dataset cache
(``lma_effort.pkl``) on first run if they are not already present.
"""

import csv
import json
import statistics
import sys
import zipfile
from pathlib import Path

import torch

from source.dataset import MotionDataset, build_lma_effort_dataset
from source.forward_kinematics import sixd_to_matrix
from source.interpolator import Interpolator
from source.synthesizer import Synthesizer

# --------------------------------------------------------------------------
# Fixed configuration (no command-line parameters by design).
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "extracted_fingerless"
PKL = ROOT / "data" / "lma_effort.pkl"
CKPT_DIR = ROOT / "checkpoints"
INTERP_W = CKPT_DIR / "interpolator.pth"
SYNTH_W = CKPT_DIR / "synthesizer.pth"
OUT = ROOT / "results"

DEVICE = "cpu"          # CPU for portability and run-to-run determinism
SEED = 0
N_BASE = 24             # number of base motions to average the effect over
NUM_JOINTS = 22
SEQ_LEN = 50
LOW, HIGH = 0.1, 0.9    # conditioned low / high descriptor values (in [0, 1])

# (label, LabanDescriptors method, index in the [V, H, P, R] style vector)
DESCRIPTORS = [("V", "vertical", 0), ("H", "horizontal", 1), ("P", "pace", 2)]


def ensure_data() -> None:
    """Check checkpoints, then extract the dataset zip and build the cache."""
    for name, path in (("interpolator.pth", INTERP_W), ("synthesizer.pth", SYNTH_W)):
        if not path.exists():
            sys.exit(f"Missing checkpoint {name}: place it in {CKPT_DIR} "
                     "(see checkpoints/README.md).")
    if not DATA_DIR.exists():
        zips = sorted((ROOT / "data").glob("extracted_fingerless*.zip"))
        if not zips:
            sys.exit("Dataset not found: place the LMA Effort BVH files under "
                     f"{DATA_DIR}, or the dataset zip in {ROOT / 'data'} "
                     "(see data/README.md).")
        print(f"Extracting {zips[0].name} ...")
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(ROOT / "data")
    if not PKL.exists():
        print("Building dataset cache (data/lma_effort.pkl); this runs once ...")
        build_lma_effort_dataset(save_path=PKL, dataset_root=DATA_DIR,
                                 device=DEVICE, sequence_length=SEQ_LEN,
                                 max_frame_diff=1)


def load_models():
    interp = Interpolator(seq_len=SEQ_LEN, num_points=4, num_coords=3,
                          encoder_hidden_dim=512, decoder_hidden_dim=512,
                          latent_dim=64, num_style_descriptors=4).to(DEVICE)
    interp.load_state_dict(torch.load(INTERP_W, map_location=DEVICE))
    interp.eval()

    synth = Synthesizer(angles_dim=NUM_JOINTS * 6, positions_dim=4 * 3,
                        conditions_dim=4).to(DEVICE)
    synth.load_state_dict(torch.load(SYNTH_W, map_location=DEVICE))
    synth.eval()
    return interp, synth


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ensure_data()
    torch.manual_seed(SEED)

    ds = MotionDataset.load(PKL, device=DEVICE)
    fk = ds.converter
    laban = ds.laban
    lim = {k: (float(v[0]), float(v[1])) for k, v in ds.laban_limits.items()}
    interp, synth = load_models()

    def normalize(value: torch.Tensor, key: str) -> torch.Tensor:
        lo, hi = lim[key]
        return (value - lo) / (hi - lo)

    @torch.no_grad()
    def pipeline(sites: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Full pipeline: sparse keyframes + style -> full-body joint positions.

        ``sites`` is (SEQ_LEN, 4, 3); ``style`` is (4,) in [0, 1]. The CVAE
        latent is reseeded identically for every call so that LOW and HIGH
        differ only in the conditioning, not in the sampled latent.
        """
        x = sites.unsqueeze(0).to(DEVICE)                       # (1, T, 4, 3)
        condition = torch.cat([x[:, 0].reshape(1, -1),
                               x[:, -1].reshape(1, -1)], dim=1)  # (1, 24)
        st = style.unsqueeze(0).to(DEVICE)                      # (1, 4)
        torch.manual_seed(SEED)
        dense = interp.generate(condition, st, device=DEVICE)   # (1, T, 4, 3)
        dense_mz = dense - dense.mean(dim=-2, keepdim=True)     # mean-center sites
        pred_6d = synth(dense_mz.reshape(1, SEQ_LEN, -1), st)
        pred_6d = pred_6d.reshape(1, SEQ_LEN, NUM_JOINTS, 6)
        positions, _ = fk.compute(sixd_to_matrix(pred_6d),
                                  torch.zeros(1, SEQ_LEN, 3, device=DEVICE))
        return positions                                        # (1, T, 22, 3)

    base_indices = torch.linspace(0, len(ds) - 1, N_BASE).long().tolist()
    base_style = torch.full((4,), 0.5)
    results = {}

    for label, method, idx in DESCRIPTORS:
        key = label.lower()
        measure = getattr(laban, method)
        lows, highs = [], []
        example = {}
        for n, bi in enumerate(base_indices):
            sites = ds.training_data["positions_sites"][bi]
            s_low = base_style.clone(); s_low[idx] = LOW
            s_high = base_style.clone(); s_high[idx] = HIGH
            pos_low = pipeline(sites, s_low)
            pos_high = pipeline(sites, s_high)
            lows.append(normalize(measure(pos_low), key).item())
            highs.append(normalize(measure(pos_high), key).item())
            if n == 0:  # keep the first base motion as the exported example
                example = {"low": pos_low[0].cpu(), "high": pos_high[0].cpu()}

        lo_mean, hi_mean = statistics.mean(lows), statistics.mean(highs)
        results[label] = {
            "conditioned_low": LOW, "conditioned_high": HIGH,
            "measured_low_mean": lo_mean, "measured_high_mean": hi_mean,
            "delta": hi_mean - lo_mean, "n_base": N_BASE,
            "effect_correct": hi_mean > lo_mean,
        }
        for which, pos in example.items():
            with open(OUT / f"motion_{label}_{which}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame", "joint", "x", "y", "z"])
                for t in range(pos.shape[0]):
                    for j in range(pos.shape[1]):
                        x, y, z = pos[t, j].tolist()
                        w.writerow([t, j, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    # ---- write summary -----------------------------------------------------
    lines = [
        "LMA descriptor-effect reproduction (paper Figs 4-8)",
        f"Base motions averaged: {N_BASE} | conditioned low={LOW} high={HIGH}",
        "",
        f"{'Descriptor':<12}{'measured(low)':>15}{'measured(high)':>16}"
        f"{'delta':>10}{'effect':>10}",
    ]
    for label, r in results.items():
        ok = "OK" if r["effect_correct"] else "FAIL"
        lines.append(f"{label:<12}{r['measured_low_mean']:>15.4f}"
                     f"{r['measured_high_mean']:>16.4f}{r['delta']:>10.4f}{ok:>10}")
    lines += ["",
              "Interpretation: for each descriptor, conditioning the pipeline on a",
              "HIGH value yields a higher measured descriptor than a LOW value",
              "(positive delta) -- the systematic style change shown in Figs 5-7.",
              "Per-example generated joint positions: results/motion_<D>_<low|high>.csv"]
    summary = "\n".join(lines)
    (OUT / "descriptor_effect.txt").write_text(summary + "\n")
    (OUT / "descriptor_effect.json").write_text(json.dumps(results, indent=2))
    print("\n" + summary)
    print(f"\nWrote results to {OUT}/")


if __name__ == "__main__":
    main()
