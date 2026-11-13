#!/usr/bin/env python3
"""Paper 1 V9 Stage 2: descriptive bilateral/intermuscular coupling from raw EMG.

Main: unrectified BR_R--BR_L early vs late disjoint 3-second windows,
plus 12-point descriptive trajectory. All 6 channel pairs and rectification
are saved as sensitivity. Three non-equivalent outcomes:
  (1) cross-wavelet relative common power (not proof of coupling)
  (2) squared Morlet wavelet coherence (WTC)
  (3) magnitude-squared Welch EMG--EMG coherence.

DO NOT infer neural origin or physiological drive from positive results.
No surrogate tests in this stage; spectral-shape and cross-talk controls are next.

WTC smoothing is a direct mathematical port of smoothwavelet from the original
MIT-licensed simulator v2.0.0, bundled as third_party/simulator_v2/functions.py.
CWT uses the L2 Morlet FFT kernels from the V9 Stage 1 implementation; this
is NOT claimed to be numerically identical to the simulator's pycwt CWT.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import coherence, fftconvolve, convolve2d
from scipy.stats import wilcoxon
from scipy.interpolate import interp1d
from frozen_stage1_reader import (load_emg,preprocess,interval_indices,hash_path,morlet_kernel)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CHANNELS=('BR_R','FCU_R','BR_L','FCU_L')
PAIRS=tuple(itertools.combinations(CHANNELS,2))
MAIN_PAIR=('BR_R','BR_L')
BANDS=((13.,30.,'beta_13_30'),(30.,50.,'low_gamma_30_50'),(30.,60.,'piper_30_60_sensitivity'))
FREQS=np.geomspace(10.,100.,28)  # exact 10-100 Hz with quasi-1/8 octave spacing
SCALES=(6+np.sqrt(38))/(4*np.pi*FREQS)
FS=1000.0
DJ=float(np.log2(FREQS[1]/FREQS[0]))
KERNELS=[morlet_kernel(float(f),FS) for f in FREQS]


def smoothwavelet_v2(wave,dt,scale,dj=DJ):
    """Same smoothing formula as simulator v2.0.0, without importing pycwt."""
    wave=np.asarray(wave)
    if wave.ndim!=2:raise ValueError('wave must be (freq,time)')
    n=wave.shape[1]
    npad=2**int(np.ceil(np.log2(n)))
    k=np.arange(1,npad//2+1)*(2*np.pi/npad)
    k=np.concatenate(([0.],k,-k[-2::-1]))
    F=np.exp(-.5*(np.asarray(scale)[:,None]/dt)**2*k[None,:]**2)
    sm=np.fft.ifft(F*np.fft.fft(wave,npad,axis=1),axis=1)[:,:n]
    if np.isrealobj(wave):sm=sm.real
    dj0steps=.6/(dj*2)
    box=np.r_[dj0steps%1,np.ones(int(2*round(dj0steps)-1)),dj0steps%1]
    box=box/box.sum()
    return convolve2d(sm,box[:,None],mode='same')


def cwt_block(sig):
    sig=np.asarray(sig,float)
    if sig.size!=3000:raise ValueError('expected 3000 samples')
    sig=sig-sig.mean()
    pad=int(FS)
    padded=np.pad(sig,(pad,pad),mode='reflect')
    arr=np.empty((len(FREQS),3000),dtype=np.complex64)
    for i,k in enumerate(KERNELS):
        arr[i]=fftconvolve(padded,k,mode='same')[pad:-pad]
    return arr


def band_average(freq,values,lo,hi):
    """Linear-frequency weighted band average, including interpolated edges."""
    freq=np.asarray(freq,float);values=np.asarray(values,float)
    vals=interp1d(freq,values,axis=0,bounds_error=False,fill_value='extrapolate')
    grid=np.r_[lo,freq[(freq>lo)&(freq<hi)],hi]
    integ=np.trapezoid(vals(grid),grid,axis=0)
    return np.asarray(integ/(hi-lo))


def metrics(x,y,cx,cy):
    sc=SCALES
    px=np.abs(cx)**2;py=np.abs(cy)**2
    cross=cx*np.conj(cy)
    sx=smoothwavelet_v2(px/sc[:,None],1/FS,sc)
    sy=smoothwavelet_v2(py/sc[:,None],1/FS,sc)
    sxy=smoothwavelet_v2(cross/sc[:,None],1/FS,sc)
    with np.errstate(divide='ignore',invalid='ignore'):
        rsq=np.clip(np.real(np.abs(sxy)**2/(sx*sy)),0,1)
    rsq[~np.isfinite(rsq)]=np.nan
    xw=np.abs(cross)/sc[:,None]
    # Conservative finite CWT support and boundary smoothing guard.
    interior=np.minimum(np.arange(3000),2999-np.arange(3000))/FS
    coi=np.maximum(.2,2*sc)
    valid=interior[None,:]>=coi[:,None]
    xwmean=np.nanmean(np.where(valid,xw,np.nan),axis=1)
    wf,coh=coherence(x,y,fs=FS,nperseg=512,noverlap=256,detrend='constant')
    result=[]
    for low,high,label in BANDS:
        cross_band=float(band_average(FREQS,xwmean,low,high))
        cross_total=float(np.trapezoid(xwmean,FREQS))
        # Total normalized to 10..100 Hz; these percentages differ from V9 Stage 1 (10..300).
        rel_xw=float(100*cross_band*(high-low)/(cross_total+1e-15))
        mask=(FREQS>=low)&(FREQS<high)
        coh_vals=np.where(valid & mask[:,None],rsq,np.nan)
        wtc_median=float(np.nanmedian(coh_vals))
        wtc_mean=float(np.nanmean(coh_vals))
        welch_mean=float(band_average(wf,coh,low,high))
        result.append(dict(band=label,coh_welch_mean=welch_mean,wavelet_coh_median=wtc_median,
           wavelet_coh_mean=wtc_mean,cross_wavelet_rel_pct=rel_xw,
           cross_wavelet_abs_au=cross_band*(high-low)))
    return result


def read_frozen():
    frozen=pd.read_csv(ROOT/'input_frozen'/'V7_emg_activity_QA.csv',encoding='utf-8-sig')
    h=pd.read_csv(ROOT/'input_frozen'/'V7_raw_EMG_sha256.csv',encoding='utf-8-sig')
    qc=pd.read_csv(ROOT/'input_frozen'/'V9_Etapa0_01_channel_QC.csv',encoding='utf-8-sig')
    assert len(frozen)==30 and len(h)==32 and len(qc)==120
    assert not frozen.force_ID.duplicated().any()
    assert not qc.duplicated(['force_ID','channel']).any()
    return frozen,dict(zip(h.force_ID,h.sha256)),{
        (r.force_ID,r.channel):('' if pd.isna(r.qc_flags) else str(r.qc_flags)) for r in qc.itertuples(index=False)}


def paired_summary(pair_df):
    main=pair_df[(pair_df.channel_A==MAIN_PAIR[0])&(pair_df.channel_B==MAIN_PAIR[1])&
                 (pair_df.modality=='interference')&(pair_df.position.isin([1,12]))&(~pair_df.flag_low_snr)]
    results=[]
    for band in [b[2] for b in BANDS]:
        a=main[(main.band==band)&(main.position==1)].set_index('force_ID')
        z=main[(main.band==band)&(main.position==12)].set_index('force_ID')
        ids=a.index.intersection(z.index)
        for measure in ['coh_welch_mean','wavelet_coh_median','cross_wavelet_rel_pct']:
            v=z.loc[ids,measure]-a.loc[ids,measure]
            results.append(dict(band=band,measure=measure,n=len(ids),median_early=float(np.median(a.loc[ids,measure])),
                median_late=float(np.median(z.loc[ids,measure])),median_paired_change=float(np.median(v)),
                n_increase=int((v>0).sum()),exploratory_wilcoxon_p=float(wilcoxon(v).pvalue)))
    return pd.DataFrame(results)


def plot_primary(pair_df,out):
    p=pair_df[(pair_df.channel_A==MAIN_PAIR[0])&(pair_df.channel_B==MAIN_PAIR[1])&(pair_df.modality=='interference')]
    fig,ax=plt.subplots(3,2,figsize=(11,9),layout='constrained')
    for j,b in enumerate(['beta_13_30','low_gamma_30_50']):
        q=p[p.band==b]
        g=q.groupby('position').agg(wtc=('wavelet_coh_median','median'),welch=('coh_welch_mean','median'),xw=('cross_wavelet_rel_pct','median'))
        xx=np.linspace(0,100,len(g))
        ax[0,j].plot(xx,g.wtc,'o-');ax[0,j].set(ylabel='Wavelet squared coherence',title=b)
        ax[1,j].plot(xx,g.welch,'o-');ax[1,j].set(ylabel='Welch magnitude-squared coherence')
        ax[2,j].plot(xx,g.xw,'o-');ax[2,j].set(ylabel='Cross-wavelet relative power (%)',xlabel='Relative EMG time (%)')
    fig.suptitle('BR_R vs BR_L | descriptive only, not a neural-source test',size=13)
    fig.savefig(out/'Figure_V9_Etapa2_BR_Bilateral_Exploratory.png',dpi=145)
    plt.close(fig)


def save(df,path):df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.10g')

def run(args):
    raw=Path(args.raw);emg_dir=raw/'EMG'
    if args.demo and not emg_dir.is_dir() and (raw/f'{args.demo}_TF90_PRE_EMG.txt').exists():emg_dir=raw
    if not emg_dir.is_dir():raise FileNotFoundError('Expected raw EMG folder')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'PROCESSING_IN_PROGRESS.txt').write_text('RUNNING\n')
    frozen,hashes,qc=read_frozen()
    if args.demo:
        frozen=frozen[frozen.force_ID==args.demo]
        if len(frozen)!=1:raise ValueError('Demo ID not in frozen cohort')
    else:
        ids={p.stem.removesuffix('_TF90_PRE_EMG') for p in emg_dir.glob('*_TF90_PRE_EMG.txt')}
        if ids!=set(hashes):raise ValueError(f'EMG raw IDs changed. expected 32, got {len(ids)}')
    rows=[];provenance=[];beg=time.monotonic()
    for idx,r in enumerate(frozen.itertuples(index=False),1):
        pid=r.force_ID;path=emg_dir/f'{pid}_TF90_PRE_EMG.txt'
        sha=hash_path(path)
        if sha!=hashes[pid]:raise ValueError(f'Raw SHA-256 mismatch {pid}')
        t,data,fs=load_emg(path)
        starts,n,limits=interval_indices(t,r.emg_active_start_s,r.emg_active_end_s,fs)
        data=np.stack([preprocess(data[:,k],fs) for k in range(4)],axis=-1)
        for modality in ('interference','rectified'):
            positions=(1,12) if modality=='rectified' else range(1,13)
            for pos in positions:
                if modality=='interference' and pos not in (1,12):
                    pairs=(MAIN_PAIR,)
                else:
                    pairs=PAIRS
                active=set(ch for pair in pairs for ch in pair)
                seg={ch:data[starts[pos-1]:starts[pos-1]+n,k] for k,ch in enumerate(CHANNELS) if ch in active}
                if modality=='rectified':seg={ch:np.abs(v) for ch,v in seg.items()}
                cwt={ch:cwt_block(v) for ch,v in seg.items()}
                for ca,cb in pairs:
                    fa=qc[pid,ca];fb=qc[pid,cb]
                    low=('LOW_SNR_REVIEW' in fa) or ('LOW_SNR_REVIEW' in fb)
                    mains=('60HZ_PEAK_REVIEW' in fa) or ('60HZ_PEAK_REVIEW' in fb)
                    for d in metrics(seg[ca],seg[cb],cwt[ca],cwt[cb]):
                        rows.append(dict(force_ID=pid,channel_A=ca,channel_B=cb,position=pos,modality=modality,
                         time_normalized_pct=100*(pos-1)/11,start_emg_s=float(t[starts[pos-1]]),
                         end_emg_s=float(t[starts[pos-1]+n-1]),flag_low_snr=low,flag_60hz=mains,
                         raw_sha256=sha,**d))
        provenance.append(dict(force_ID=pid,sha256=sha,start_s=r.emg_active_start_s,end_s=r.emg_active_end_s))
        print(f'[{idx}/{len(frozen)}] {pid}: 6 pair early/late + BR bilateral trajectory',flush=True)
    df=pd.DataFrame(rows)
    save(df,out/'01_pair_wavelet_and_welch_metrics.csv')
    save(pd.DataFrame(provenance),out/'02_provenance.csv')
    if not args.demo:
        tests=paired_summary(df);save(tests,out/'03_BR_bilateral_exploratory_paired.csv')
        plot_primary(df,out)
    else:
        tests=paired_summary(df);save(tests,out/'03_BR_bilateral_exploratory_paired.csv')
    result=dict(status='PASS',demo=bool(args.demo),participants=df.force_ID.nunique(),rows=len(df),
      independent_unit='participant',primary_pair='BR_R vs BR_L',
      no_force_emg_synchronization=True,
      no_neural_drive_inference=True,
      warnings=['no surrogate-matched null yet','no crosstalk correction','3-s windows may overlap for descriptive trajectory',
        'cross-wavelet relative power denominator is 10-100 Hz unlike V9 Stage1 10-300 Hz',
        'CWT kernels from V9 Stage1; smoothing from simulator v2.0.0; not bitwise identical to pycwt'],
      wall_time_s=round(time.monotonic()-beg,1))
    (out/'04_SUMMARY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'PROCESSING_IN_PROGRESS.txt').unlink()
    print(json.dumps(result,indent=2))
    return result


def test_synthetic():
    rng=np.random.default_rng(2109)
    t=np.arange(3000)/FS
    u=np.sin(2*np.pi*20*t)
    a=u+.8*rng.standard_normal(3000)
    b=u+.8*rng.standard_normal(3000)
    c=rng.standard_normal(3000)
    m_shared=metrics(a,b,cwt_block(a),cwt_block(b))
    m_indep=metrics(a,c,cwt_block(a),cwt_block(c))
    ba=next(x for x in m_shared if x['band']=='beta_13_30')
    bi=next(x for x in m_indep if x['band']=='beta_13_30')
    print('Synthetic common 20 Hz:',ba)
    print('Synthetic independent:',bi)
    assert ba['coh_welch_mean']>bi['coh_welch_mean']+.15
    assert ba['wavelet_coh_median']>bi['wavelet_coh_median']+.15
    print('SYNTHETIC_PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw',default='private_data_NOT_DISTRIBUTED')
    ap.add_argument('--out',default=None)
    ap.add_argument('--demo',default=None)
    ap.add_argument('--test-synthetic',action='store_true')
    a=ap.parse_args()
    if a.test_synthetic:test_synthetic()
    else:
        if a.out is None:a.out=str(Path(a.raw).parent/('Paper1_V9_Etapa2_TESTE_ABC' if a.demo else 'Paper1_V9_Etapa2_Resultados'))
        run(a)
