# Reproduction claims and boundaries

## Available publicly

**Stage 4**: the nine-mode bilateral simulation, with seed rule `2026110904 + 2003*i`, original model parameters and 32 independent realizations. Executing `stage4/simulate_stage4.py --subjects 32` regenerates simulation-only tables and internal controls. The code checks spike-train identity in slowing contrasts and amplitude-only power scaling and coherence invariance. The reference aggregate `results/simulation_only/04_GROUP_SUMMARY.csv` was copied from archived simulation outputs, not reestimated from human EMG.

**V7f**: simulation engine and analysis code provided; `examples/demo_v7f.py` runs a simulation-only example. The exact V7f paper comparator checks a participant-derived table with a SHA256 fingerprint, which is intentionally excluded from this public repository. The separate pre-existing simulation baseline source is also required for one historical consistency check. Thus the full original V7f comparison is **not self-contained** in the public package. It was previously run in the authors' private reproducibility environment.

## Not publicly reproducible from this archive alone

All analyses of participant-level force/EMG signals (including V9 stages 0–3b) require private recordings. An author can rerun and compare those analyses privately. An independent reader cannot reproduce human numerical outcomes solely from the public source archive.

## Interpretive safeguards

- Baseline shared oscillatory drive is an input condition in the bilateral simulator; it is not evidence that an observed human signal came from cortex.
- `sensor_crosstalk_10pct` is a mathematical mixture: L'=L+0.10 R, R'=R+0.10 L. This is **not** 10% shared input, 10% power, or direct muscular cross-talk between distant forearms.
- Coupling power, normalized coherence and motor-unit common input are distinct measures.
- The simulations are not calibrated to individual participants, and numerical sensitivity controls are not measurements of the source of neural drive in participants.

## Portability modification

Two hardcoded private Windows raw-data default paths were removed from `stage4/modules` for public distribution. These defaults are **not used** in the Stage 4 simulation (`preprocess`, `welch_mdf`, `cwt_block`, `metrics`). No simulated numerical method, seed, filter, or parameter was changed. The original private sources should remain in the reproducibility master.
