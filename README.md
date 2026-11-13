# Surface EMG frequency rescaling and bilateral motor-unit simulation — Paper 1

This package contains simulation methods used with the manuscript *Spectral compression and shape change in surface EMG at task failure: a frequency-rescaling framework tested with motor-unit simulation*.

The code demonstrates that changes in motor-unit action-potential propagation, signal amplitude, and imposed shared oscillatory input can affect spectral and bilateral coupling measures. **Simulated shared input is known by construction; the study does not identify a cortical, spinal, or subcortical source in human recordings.** Artificial bidirectional channel mixing is a **mathematical estimator sensitivity test**, not a plausible direct muscular cross-talk mechanism between opposite forearms.

**Archived software release:** [v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.24187981) (DOI: `10.5281/zenodo.24187981`). [GitHub release v1.0.0](https://github.com/osmar235/paper1-emg-spectral-simulations/releases/tag/v1.0.0). The separately published underlying simulator has a different DOI (`10.5281/zenodo.19703513`).

## Included

- `stage4/`: original bilateral 32-realization Stage 4 simulation, including methods and mathematical checks. Source is unchanged except that two unused default Windows paths were replaced with a generic placeholder in `modules/` for privacy.
- `v7f/`: V7f source, conduction slowing and stochastic-stream controls. The complete archived human comparison requires a **private participant-derived reference table** not distributed here. A public V7f *simulation-only* example is provided.
- `results/simulation_only/04_GROUP_SUMMARY.csv`: summary from the archived 32-realization bilateral simulation, **no human observations**.
- `examples/`: quick starts that create only synthetic simulation outputs.
- `docs/`: interpretation, reproducibility, data access, release instructions.

No original or participant-level human EMG or force data are included. This code **does not** reproduce the human results without the restricted data.

## Install (Windows PowerShell / macOS / Linux)

Python 3.11+ (tested here with Python 3.13.5 and NumPy 2.x). In a fresh directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Example A — one bilateral simulation realization (not publication inference)

```bash
python stage4/simulate_stage4.py --subjects 1 --out outputs/stage4_demo
```

This writes `01_MECHANISM_QA.csv`, `02_ALL_METRICS.csv`, `03_PAIRED_EFFECTS.csv`, `04_GROUP_SUMMARY.csv`, a control figure, and `05_VERIFICATION.json`. A one-realization run is labeled **QA_ONLY**, not a replication of group inference.

### Example B — full bilateral simulation (reproducibility mode)

```bash
python stage4/simulate_stage4.py --subjects 32 --out outputs/stage4_full
```

Use default parameters and the same Python/scientific-library versions when comparing with the original archived simulation. Small numeric differences can arise across environments; compare within documented tolerance rather than demanding bitwise equality. The code carries its own physiological invariant tests (spike-train matching and amplitude-only control).

### Example C — public single-muscle motor-unit example

```bash
python examples/demo_v7f.py
```

This runs the published single-pool simulation kernel with two conduction settings and reports spectral differences. It does **not** use human data or recreate the full 32-realization V7f results.

## Limits

This simulation code illustrates numerical and physiological mechanisms under specified models. It does not localize common neural drive and is not a validated estimator of neural contribution from a single pair of interference EMG channels. Human force and EMG recordings may be requested from the corresponding author subject to consent and institutional/ethics approval, and are not bundled or publicly deposited.

## Citation and rights

- Underlying simulator: Pinto Neto et al. (2026), *Computer Methods and Programs in Biomedicine*, DOI: https://doi.org/10.1016/j.cmpb.2026.109657; underlying software https://doi.org/10.5281/zenodo.19703513.
- This Paper 1 code: https://github.com/osmar235/paper1-emg-spectral-simulations. Add the **new** Zenodo version DOI to `CITATION.cff` after Zenodo archives the GitHub release. Do **not** confuse it with the already-published simulator DOI.
- The reused `simulator/functions.py` is MIT-licensed and its original license and authors are preserved. The additional Paper 1 code is released under the MIT License in the repository root, subject to the rights of each contributor. Retain the separate original MIT licenses and copyright notices of the reused simulator components.

Reproducibility details: `docs/REPRODUCIBILITY.md`. Release workflow: `docs/GITHUB_ZENODO_RELEASE.md`.