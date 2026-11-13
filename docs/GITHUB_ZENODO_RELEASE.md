# Publication procedure — GitHub + Zenodo

This repository contains the author-approved MIT release candidate. Complete preflight checks and preserve the upstream licenses before tagging a Zenodo release.

1. Confirm the approved root MIT license and each contributor’s permission to publish their work. The reused simulator retains its separate MIT notices.
2. Check all files in this **separate clean directory** for personal participant data, credentials, private local paths, institution-only files, and private review notes. Never `git add .` from the private master folder.
3. Run in a clean Python environment: `python -m pip install -r requirements.txt`, `python examples/demo_v7f.py`, and `python stage4/simulate_stage4.py --subjects 1 --out outputs/stage4_demo`. Save the resulting QA PASS evidence outside the Git-tracked source folder.
4. Use the GitHub repository https://github.com/osmar235/paper1-emg-spectral-simulations. Upload the clean source files, not the original private master or nested ZIP archives.
5. Link GitHub to Zenodo through Zenodo account settings, select the repo under Zenodo > GitHub, click Sync and enable the repository (see official guide https://help.zenodo.org/docs/github/enable-repository/).
6. Create the GitHub release with tag `v1.0.0`. Zenodo ingests the release and registers a **new** DOI. Do not add a made-up DOI in advance. (Official guide https://help.zenodo.org/docs/github/archive-software/github-upload/.)
7. Capture exact GitHub URL, release tag, **version DOI**, and release date. Update `CITATION.cff`, manuscript *Data and code availability*, and the final deposition checklist. Record a SHA256 fingerprint of the uploaded ZIP.
8. Add citation and DOI in the manuscript only **after** the public record exists and is working. Do not describe the full human analysis as independently reproducible from this software-only deposit.

**Privacy**: Raw human data, individual-level processed data and identifiable metadata are *not* publicly shared. Requests must undergo a real case-by-case consent/ethics/institutional assessment. Offering data on request is not equivalent to authorizing distribution. Review whether the JEK portal requires additional data statement fields.

**Zenodo archival policy**: a published Zenodo record's files cannot simply be replaced; use new versions for updates. Coordinate the tag, license, authors and date before the public release.
