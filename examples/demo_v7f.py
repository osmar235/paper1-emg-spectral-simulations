#!/usr/bin/env python3
"""Simulation-only V7f demonstration. NO HUMAN EMG OR HUMAN COMPARATOR."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'v7f/source'))
import tf_emg_sim as sim
import spectral_tools as spectra

def main():
    sim.load_simulator(str(ROOT/'v7f/simulator'))
    conf=dict(duration_s=3.0,seed=20261108,fs_out=1000.0,n_mu=12,dt=.0005,n_fibers=64,burn_in_s=.5,dur_rank_link=.9,return_spikes=True)
    early,info0=sim.simulate_tf_segment(cv_scale=1.0,**conf)
    late,info1=sim.simulate_tf_segment(cv_scale=1.2,**conf)
    assert early.size==late.size==3000, 'Invalid 3 s simulated segment'
    assert np.array_equal(info0['spikes'],info1['spikes']), 'Discharge train changed with conduction-only manipulation'
    p0=spectra.denotch(spectra.psd(early));p1=spectra.denotch(spectra.psd(late))
    m0=spectra.mdf(p0);m1=spectra.mdf(p1)
    assert np.isfinite([m0,m1]).all()
    print('PASS: simulation-only demonstration; fixed motor-unit discharges')
    print(f'Simulated early MDF: {m0:.3f} Hz; late MDF: {m1:.3f} Hz')
    print('No human-derived reference or inferential comparison is included.')

if __name__=='__main__':main()
