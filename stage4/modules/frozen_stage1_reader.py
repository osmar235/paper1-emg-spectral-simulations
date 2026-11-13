#!/usr/bin/env python3
"""Paper 1 V9 stage 1: Morlet (omega_0=6) band power and MDF-based compression prediction.

FROZEN design: 30 EMG participants from V7, 4 channels, 3-second windows at 12
normalized positions. Two non-overlapping 3 s windows are the only primary contrast.
Interference EMG primary; rectification secondary. No force/EMG clock synchrony is assumed.

Primary estimated quantity per channel/position:
    observed relative 13-30/30-50 Hz wavelet energy percentage (within 10-300 Hz)
    vs baseline-Morlet spectrum compressed by that channel's interference MDF ratio.
The residual is an observed-minus-predicted band-PERCENTAGE-POINT difference,
not 'neural drive' nor a statistical variance explained by physiology.

The Morlet CWT is implemented using L2-normalized complex wavelets (w0=6)
via scipy FFT convolution with reflected padding. The user-supplied original
simulator v2.0.0 is bundled separately for subsequent coherence work; this
implementation is NOT claimed to be bitwise identical to the 2010 MATLAB routines.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt,iirnotch,welch,fftconvolve
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RAW_DEFAULT=Path('private_data_NOT_DISTRIBUTED')
CHANNELS=('BR_R','FCU_R','BR_L','FCU_L')
PRIMARY_CHANNELS=('BR_R','FCU_R','BR_L')
BANDS=((13.,30.,'beta_13_30'),(30.,50.,'low_gamma_30_50'),(30.,60.,'piper_30_60_sensitivity'))
FMIN,FMAX=10.,300.
FS=1000.
W0=6.
N_POSITIONS=12
WINDOW_S=3.
GRID=np.unique(np.r_[np.geomspace(10,450,100),10.,13.,30.,50.,60.,300.,450.]); GRID.sort()


def hash_path(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()


def load_emg(p):
 a=np.loadtxt(io.BytesIO(p.read_bytes().replace(b',',b'.')),delimiter='\t',usecols=(0,1,2,3,4),dtype=float)
 if a.ndim!=2 or a.shape[1]!=5 or not np.all(np.isfinite(a)) or len(a)<5000:raise ValueError(f'Invalid EMG matrix: {p}')
 t=a[:,0];dt=np.diff(t);fs=1/np.median(dt)
 if not np.isclose(fs,1000,atol=10) or np.any(dt<=0) or np.max(np.abs(dt-np.median(dt)))>.006:
  raise ValueError(f'Unexpected EMG timestamps: {p}; fs={fs}')
 return t,a[:,1:].astype(np.float64),float(fs)


def preprocess(x,fs):
 y=np.asarray(x,float).copy()
 for freq in (60.,120.,180.):
  b,a=iirnotch(freq,30,fs);y=filtfilt(b,a,y)
 b,a=butter(4,[10/(fs/2),450/(fs/2)],btype='band');return filtfilt(b,a,y)


def interval_indices(t,on,off,fs):
 a=int(np.searchsorted(t,float(on),side='left'));b=int(np.searchsorted(t,float(off),side='left'))
 start=a+int(round(1.5*fs));end=b-int(round(.5*fs));n=int(round(WINDOW_S*fs))
 if start<0 or end>len(t) or end-start<2*n:
  raise ValueError(f'V7 active span does not permit disjoint 3 s windows: {on}, {off}')
 starts=np.rint(np.linspace(start,end-n,N_POSITIONS)).astype(int)
 if starts[0]!=start or starts[-1]!=end-n or np.any(np.diff(starts)<=0):
  raise ValueError('Invalid 12-position placement')
 return starts,n,(a,b,start,end)


def morlet_kernel(f,fs):
 # Fourier period of Morlet w0=6 according to Torrence & Compo: period=4*pi*s/(w0+sqrt(2+w0²)).
 # f = (w0+sqrt(2+w0²))/(4*pi*s); s in seconds.
 s=(W0+np.sqrt(W0*W0+2))/(4*np.pi*f)
 times=np.arange(-int(np.ceil(5*s*fs)),int(np.ceil(5*s*fs))+1)/fs
 k=np.exp(-0.5*(times/s)**2)*np.exp(1j*W0*times/s)
 k=k/np.sqrt(np.sum(np.abs(k)**2))
 return k

KERNELS=[morlet_kernel(float(f),FS) for f in GRID]


def cwt_mean_power(seg,fs,gr=GRID):
 # 1 s reflection margin avoids treating a 3 s window edge as a zero-activity event.
 if abs(fs-FS)>1e-6:raise ValueError('Morlet kernels fixed at fs=1000')
 x=np.asarray(seg,float)
 if len(x)!=3000:raise ValueError('Expected exactly 3000 samples')
 x=x-np.mean(x)
 n_pad=int(fs*1.)
 xp=np.pad(x,(n_pad,n_pad),mode='reflect')
 out=np.empty(len(gr),dtype=float)
 for i,k in enumerate(KERNELS):
  z=fftconvolve(xp,k,mode='same')[n_pad:-n_pad]
  out[i]=np.mean(np.abs(z)**2)
 return np.maximum(out,0)


def denotch_spectrum(p,frequencies=GRID):
 y=np.array(p,copy=True,dtype=float)
 for center in (60,120,180):
  gap=(frequencies>center-6)&(frequencies<center+6)
  if gap.any():y[gap]=np.interp(frequencies[gap],frequencies[~gap],y[~gap])
 return np.maximum(y,0)


def integ(q,lo,hi,grid=GRID):
 a=np.r_[lo,grid[(grid>lo)&(grid<hi)],hi]
 return float(np.trapezoid(np.interp(a,grid,q),a))


def band_values(raw,demod):
 total=integ(demod,FMIN,FMAX)
 if not np.isfinite(total) or total<1e-12:raise ValueError('Near-zero Morlet total in 10-300 Hz')
 d={}
 for lo,hi,key in BANDS:
  d[key+'_rel_pct']=100*integ(demod,lo,hi)/total
  d[key+'_abs_au']=integ(raw,lo,hi)
  d[key+'_denotch_abs_au']=integ(demod,lo,hi)
 d['total_abs_10_300_au']=integ(raw,FMIN,FMAX)
 d['total_denotch_10_300_au']=total
 return d


def welch_mdf(seg,fs):
 f,p=welch(seg,fs=fs,nperseg=512,noverlap=256,detrend='constant')
 m=(f>=10)&(f<=300);f=f[m];p=p[m]
 good=np.ones(len(f),bool)
 for center in (60,120,180):
  bad=(f>center-6)&(f<center+6)
  if bad.any():p[bad]=np.interp(f[bad],f[~bad],p[~bad])
 cf=np.cumsum(p)
 if cf[-1]<=0:raise ValueError('Zero Welch spectrum')
 return float(np.interp(.5*cf[-1],cf,f))


def compressed_power(base,ratio,grid=GRID):
 # Frequency-only compression with conservation of total spectral energy on an unbounded domain:
 # p_t(f) = p_0(f/r)/r. Re-normalize observed and predicted within identical 10-300 Hz support.
 if ratio<=0 or not np.isfinite(ratio):raise ValueError('Invalid MDF ratio')
 return np.maximum(np.interp(grid/ratio,grid,base,left=0,right=0)/ratio,0)


def save_csv(records,path):
 df=pd.DataFrame(records)
 df.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.10g')
 return df


def paired_bootstrap_median_ci(x,rng,nboot=2000):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if len(x)==0:return np.nan,np.nan
 s=np.median(x[rng.integers(0,len(x),(nboot,len(x)))],axis=1)
 return tuple(np.quantile(s,[.025,.975]).astype(float))


def test_zero(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)]
 if len(a)<5:return np.nan
 if np.max(np.abs(a))<1e-12:return 1.
 return float(wilcoxon(a,zero_method='wilcox',alternative='two-sided').pvalue)


def bh_q(pvals):
 p=np.array(pvals,float);q=np.full(len(p),np.nan);ok=np.flatnonzero(np.isfinite(p))
 if len(ok):
  order=ok[np.argsort(p[ok])];qm=np.minimum.accumulate((p[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1]
  q[order]=np.minimum(qm,1.)
 return q


def make_summary(df,out):
 # Channel-level data -> participant means: avoid counting 3+ channels as independent people.
 grp=(df[(df.modality=='interference')&(df.primary_channel_ok)&(df.channel.isin(PRIMARY_CHANNELS))]
      .groupby(['force_ID','position','band'],as_index=False).agg(obs_pct=('observed_relative_pct','mean'),pred_pct=('predicted_relative_pct','mean'),
       resid_pp=('residual_pct_points','mean'),mdf_ratio=('mdf_ratio','mean'),nchannels=('channel','nunique')))
 if grp.groupby('force_ID').nchannels.min().min()<2:raise ValueError('A participant has <2 eligible primary channels')
 if grp.force_ID.nunique()!=30 and not out.name.endswith('ABC'): # demo len=1
  raise ValueError('Expected 30 participants in group data')
 save_csv(grp.to_dict('records'),out/'02_participant_means_3channels.csv')
 trajectory=[];contrast=[]
 for band in [b[2] for b in BANDS]:
  sub=grp[grp.band==band]
  for pos,gg in sub.groupby('position'):
   d={'band':band,'position':int(pos),'relative_time_pct':float((pos-1)*100/(N_POSITIONS-1)),'n':int(gg.force_ID.nunique())}
   rng=np.random.default_rng(20261109+int(pos)+{'beta_13_30':13,'low_gamma_30_50':30,'piper_30_60_sensitivity':60}[band])
   for col in ['obs_pct','pred_pct','resid_pp','mdf_ratio']:
    d[col+'_median']=float(gg[col].median())
    d[col+'_ci_lo'],d[col+'_ci_hi']=paired_bootstrap_median_ci(gg[col].values,rng)
   trajectory.append(d)
  first=sub[sub.position==1].set_index('force_ID')
  last=sub[sub.position==12].set_index('force_ID')
  common=first.index.intersection(last.index)
  r=last.loc[common].copy();b=first.loc[common]
  delta_obs=r.obs_pct-b.obs_pct;delta_pred=r.pred_pct-b.pred_pct
  residual=delta_obs-delta_pred
  d={'band':band,'n_subjects':len(common),'n_eligible_channels_min':int(r.nchannels.min()),
    'obs_change_pp_median':float(np.median(delta_obs)),'pred_change_pp_median':float(np.median(delta_pred)),
    'residual_change_pp_median':float(np.median(residual)),
    'p_residual_vs_zero_exploratory':test_zero(residual)}
  rng=np.random.default_rng(20261109)
  d['residual_ci95_lo'],d['residual_ci95_hi']=paired_bootstrap_median_ci(residual.values,rng,5000)
  contrast.append(d)
 save_csv(trajectory,out/'03_trajectory_participant_median_ci.csv')
 dfc=save_csv(contrast,out/'04_primary_early_late_tests.csv')
 # Predeclared BH across TWO primary interference bands; 30-60 is sensitivity and has no q value.
 q=bh_q(dfc.loc[dfc.band.isin(('beta_13_30','low_gamma_30_50')),'p_residual_vs_zero_exploratory'].values)
 dfc['q_BH_two_primary_bands']=np.nan
 dfc.loc[dfc.band.isin(('beta_13_30','low_gamma_30_50')),'q_BH_two_primary_bands']=q
 dfc.to_csv(out/'04_primary_early_late_tests.csv',index=False,encoding='utf-8-sig',float_format='%.10g')
 return grp,dfc



def make_sensitivities(df,out):
    """Descriptive prespecified QC/modality sensitivity; each subject weighted equally."""
    modes={
      'interference_three_channels_lowSNR_excluded': (df.modality=='interference') & df.channel.isin(PRIMARY_CHANNELS) & (~df.flagged_low_snr),
      'interference_three_channels_no_QC_exclusion': (df.modality=='interference') & df.channel.isin(PRIMARY_CHANNELS),
      'interference_three_channels_exclude_60Hz_flags': (df.modality=='interference') & df.channel.isin(PRIMARY_CHANNELS) & (~df.flagged_low_snr) & (~df.flagged_mains60),
      'interference_BR_bilateral_QC': (df.modality=='interference') & df.channel.isin(('BR_R','BR_L')) & (~df.flagged_low_snr),
      'interference_four_channels_QC': (df.modality=='interference') & (~df.flagged_low_snr),
      'rectified_three_channels_lowSNR_excluded': (df.modality=='rectified') & df.channel.isin(PRIMARY_CHANNELS) & (~df.flagged_low_snr),
    }
    results=[]
    for mode,mask in modes.items():
      g=df.loc[mask].groupby(['force_ID','band','position'],as_index=False).agg(
        obs=('observed_relative_pct','mean'),pred=('predicted_relative_pct','mean'),res=('residual_pct_points','mean'),
        channels=('channel','nunique'))
      for band in [b[2] for b in BANDS]:
        a=g[(g.band==band)&(g.position==1)].set_index('force_ID')
        z=g[(g.band==band)&(g.position==12)].set_index('force_ID')
        ids=a.index.intersection(z.index)
        if len(ids)==0:continue
        do=z.loc[ids,'obs']-a.loc[ids,'obs'];dp=z.loc[ids,'pred']-a.loc[ids,'pred']
        results.append(dict(sensitivity=mode,band=band,n_subjects=len(ids),
          min_channels_per_subject=int(z.loc[ids,'channels'].min()),
          median_observed_change_pp=float(np.median(do)),
          median_predicted_change_pp=float(np.median(dp)),
          median_residual_pp=float(np.median(do-dp)),
          residual_positive_subjects=int(np.sum((do-dp)>0))))
    save_csv(results,out/'08_QC_rectification_sensitivity_descriptive.csv')

    # Separate channels for later analysis (not four times the independent N).
    channels=[]
    for (ch,method,band),sub in df.groupby(['channel','modality','band']):
      b=sub[sub.position==1].set_index('force_ID');l=sub[sub.position==12].set_index('force_ID')
      ids=b.index.intersection(l.index)
      good=np.array([not bool(b.loc[i,'flagged_low_snr']) for i in ids])
      used=ids[good]
      if not len(used):continue
      d=l.loc[used,'observed_relative_pct']-b.loc[used,'observed_relative_pct']
      p=l.loc[used,'predicted_relative_pct']-b.loc[used,'predicted_relative_pct']
      channels.append(dict(channel=ch,method=method,band=band,n=len(used),
       delta_observed_pp_median=float(np.median(d)),delta_predicted_pp_median=float(np.median(p)),
       delta_residual_pp_median=float(np.median(d-p))))
    save_csv(channels,out/'09_channel_level_descriptive.csv')

def make_overview(grp,out):
 fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 for i,band in enumerate(('beta_13_30','low_gamma_30_50')):
  sub=grp[grp.band==band]
  st=sub.groupby('position').agg(obs=('obs_pct','median'),pred=('pred_pct','median'),resid=('resid_pp','median'),mdf=('mdf_ratio','median'))
  tx=np.linspace(0,100,12)
  ax[0,i].plot(tx,st.obs,color='#173A5E',marker='o',label='Observed Morlet')
  ax[0,i].plot(tx,st.pred,color='#CB811E',marker='s',label='MDF compression-only')
  ax[0,i].set(title=band.replace('_',' '),xlabel='Relative EMG contraction time (%)',ylabel='Normalized wavelet power (%)');ax[0,i].legend(fontsize=8)
  ax[1,i].plot(tx,st.resid,color='#167F77',marker='o')
  ax[1,i].axhline(0,color='black',lw=.7)
  ax[1,i].set(xlabel='Relative EMG contraction time (%)',ylabel='Observed - predicted (percentage points)',title='Residual band-power trajectory')
 fig.suptitle('V9 Stage 1 - interference EMG, 30 participant-level trajectories\nExploratory normalized-time trajectories; overlapping windows',fontsize=12)
 fig.savefig(out/'Figure_V9_Etapa1_Morlet_vs_MDF.png',dpi=165);plt.close(fig)


def subject_plot(pid,df,out):
 fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 dat=df[(df.force_ID==pid)&(df.modality=='interference')]
 for i,band in enumerate(('beta_13_30','low_gamma_30_50')):
  q=dat[(dat.band==band)&(dat.channel.isin(PRIMARY_CHANNELS))&dat.primary_channel_ok].groupby('position').agg(obs=('observed_relative_pct','mean'),pred=('predicted_relative_pct','mean'),res=('residual_pct_points','mean'))
  t=np.linspace(0,100,len(q))
  axs[0,i].plot(t,q.obs,marker='o',label='Observed');axs[0,i].plot(t,q.pred,marker='s',label='MDF-only')
  axs[0,i].set(title=band,xlabel='Relative time (%)',ylabel='Power (% of 10-300 Hz)');axs[0,i].legend(fontsize=8)
  axs[1,i].plot(t,q.res,color='tab:green',marker='o');axs[1,i].axhline(0,color='black',lw=.7)
  axs[1,i].set(xlabel='Relative time (%)',ylabel='Observed - predicted (p.p.)')
 fig.suptitle(f'{pid}: primary 3-channel eligible mean (NO forced B/EMG sync)',fontsize=12)
 fig.savefig(out/'QA_figures'/f'{pid}_V9_Morlet_MDF.png',dpi=125);plt.close(fig)


def run(args):
 root=Path(args.raw);emg_dir=root/'EMG'
 if args.demo and not emg_dir.is_dir() and (root/(args.demo+'_TF90_PRE_EMG.txt')).exists():emg_dir=root
 if not emg_dir.is_dir():raise FileNotFoundError(f'Expected EMG directory {root / "EMG"}')
 out=Path(args.out)
 frozen=pd.read_csv(HERE/'input_frozen'/'V7_emg_activity_QA.csv',encoding='utf-8-sig')
 hashes=pd.read_csv(HERE/'input_frozen'/'V7_raw_EMG_sha256.csv',encoding='utf-8-sig')
 qc=pd.read_csv(HERE/'input_frozen'/'V9_Etapa0_01_channel_QC.csv',encoding='utf-8-sig')
 if len(frozen)!=30 or len(hashes)!=32 or len(qc)!=120 or qc.duplicated(['force_ID','channel']).any():raise ValueError('Frozen reference size differs; STOP')
 hash_map=dict(zip(hashes.force_ID,hashes.sha256));qc_map={(r.force_ID,r.channel):r for r in qc.itertuples(index=False)}
 if args.demo:
  if args.demo not in set(frozen.force_ID):raise ValueError('Demo participant not in frozen set')
  frozen=frozen[frozen.force_ID==args.demo]
 else:
  files={p.stem.removesuffix('_TF90_PRE_EMG') for p in emg_dir.glob('*_TF90_PRE_EMG.txt')}
  if files!=set(hashes.force_ID):raise ValueError(f'Mismatch in 32 raw EMG IDs: missing={sorted(set(hashes.force_ID)-files)} extra={sorted(files-set(hashes.force_ID))}')
 out.mkdir(parents=True,exist_ok=True)
 for old in ('06_SUMMARY.json','10_VERIFICACAO_FINAL.json'):
  (out/old).unlink(missing_ok=True)
 (out/'PROCESSING_IN_PROGRESS.txt').write_text('In progress; if interrupted, output must not be treated as final.\n',encoding='utf-8')
 (out/'QA_figures').mkdir(exist_ok=True);(out/'spectra_npz').mkdir(exist_ok=True)
 allrows=[];prov=[];t0=time.time()
 for idx,row in enumerate(frozen.itertuples(index=False),1):
  pid=row.force_ID;p=emg_dir/(pid+'_TF90_PRE_EMG.txt')
  if not p.exists():raise FileNotFoundError(str(p))
  sha=hash_path(p)
  if sha!=hash_map[pid]:raise ValueError(f'SHA mismatch for {pid}: raw file does not match frozen V7')
  t,x,fs=load_emg(p);starts,n,limits=interval_indices(t,row.emg_active_start_s,row.emg_active_end_s,fs)
  # Filter WHOLE trace before cropping (filtfilt edges outside analyzed windows).
  processed=np.column_stack([preprocess(x[:,k],fs) for k in range(4)])
  all_spectra=np.zeros((4,2,12,len(GRID)),dtype=np.float32)
  denotch=np.zeros_like(all_spectra)
  mdf_ratios=np.zeros((4,12),dtype=np.float32)
  mdl=np.zeros_like(all_spectra)
  cache=[]
  for k,ch in enumerate(CHANNELS):
   qc_row=qc_map[pid,ch];flags= str(qc_row.qc_flags) if pd.notna(qc_row.qc_flags) else ''
   low_snr=('LOW_SNR_REVIEW' in flags)
   mains=('60HZ_PEAK_REVIEW' in flags)
   primary_ok=(ch in PRIMARY_CHANNELS) and not low_snr
   MDF=[]
   for j,start in enumerate(starts):
    y=processed[start:start+n,k]
    MDF.append(welch_mdf(y,fs))
   r=np.asarray(MDF)/MDF[0];mdf_ratios[k]=r.astype(np.float32)
   for m,method in enumerate(('interference','rectified')):
    for j,start in enumerate(starts):
     y=processed[start:start+n,k]
     if method=='rectified':y=np.abs(y)
     spec=cwt_mean_power(y,fs)
     all_spectra[k,m,j]=spec
     denotch[k,m,j]=denotch_spectrum(spec)
    baseline=denotch[k,m,0].astype(float)
    for j,start in enumerate(starts):
     obs=denotch[k,m,j].astype(float)
     pred=compressed_power(baseline,r[j]);mdl[k,m,j]=pred
     bvals=band_values(all_spectra[k,m,j],obs)
     predtot=integ(pred,FMIN,FMAX)
     if predtot<=0:raise ValueError('Zero prediction')
     for lo,hi,band in BANDS:
      pred_pct=100*integ(pred,lo,hi)/predtot
      allrows.append(dict(force_ID=pid,channel=ch,modality=method,band=band,position=j+1,
       relative_time_pct=j*100/(N_POSITIONS-1),start_emg_s=float(t[start]),end_emg_s=float(t[start+n-1]),
       window_duration_s=WINDOW_S,window_overlaps_neighbors=bool((starts[1]-starts[0])<n),
       mdf_interference_Hz=float(MDF[j]),mdf_interference_baseline_Hz=float(MDF[0]),mdf_ratio=float(r[j]),
       observed_relative_pct=bvals[band+'_rel_pct'],predicted_relative_pct=pred_pct,
       residual_pct_points=bvals[band+'_rel_pct']-pred_pct,
       observed_abs_band_au=bvals[band+'_abs_au'],observed_denotch_abs_band_au=bvals[band+'_denotch_abs_au'],
       pred_abs_conserved_au=integ(pred,lo,hi),
       observed_total_abs_10_300_au=bvals['total_abs_10_300_au'],
       predicted_abs_amplitude_matched_au=integ(pred,lo,hi)*bvals['total_denotch_10_300_au']/predtot,
       flagged_low_snr=low_snr,flagged_mains60=mains,primary_channel_ok=primary_ok,
       frozen_activity_start_s=float(row.emg_active_start_s),frozen_activity_end_s=float(row.emg_active_end_s),
       raw_sha256=sha))
  np.savez_compressed(out/'spectra_npz'/f'{pid}_V9_Morlet_spectra.npz',
    frequencies_Hz=GRID,channel_names=np.array(CHANNELS),method_names=np.array(['interference','rectified']),
    raw_morlet_power=all_spectra,denotch_morlet_power=denotch,compression_model_power=mdl,
    mdf_ratio=mdf_ratios,window_start_emg_s=t[starts],window_end_emg_s=t[starts+n-1],raw_sha256=np.array(sha))
  prov.append(dict(force_ID=pid,raw_file=p.name,raw_sha256=sha,emg_activity_start_s=row.emg_active_start_s,emg_activity_end_s=row.emg_active_end_s,
   N_positions=len(starts),first_window_start_s=t[starts[0]],last_window_end_s=t[starts[-1]+n-1],no_force_emg_synchronization=True))
  print(f'[{idx}/{len(frozen)}] {pid}: processed 4 channels x 2 variants x 12 positions',flush=True)
 df=save_csv(allrows,out/'01_channel_window_band_morlet.csv')
 save_csv(prov,out/'05_provenance_verified.csv')
 # Diagnostic one row per channel/method/position with exact output frequencies in NPZ.
 grp,dfc=make_summary(df,out)
 make_sensitivities(df,out)
 if not args.demo:
  for pid in sorted(df.force_ID.unique()):subject_plot(pid,df,out)
  make_overview(grp,out)
 else:subject_plot(args.demo,df,out)
 summary=dict(status='PASS',stage='V9_1',demo=bool(args.demo),n_participants=int(df.force_ID.nunique()),
  n_channels=len(CHANNELS),n_modalities=2,n_positions=N_POSITIONS,n_bands=len(BANDS),
  n_rows_channel_window_band=int(len(df)),n_primary_paired_subjects=int(grp.force_ID.nunique()),
  methods='L2-normalized Morlet CWT w0=6, reflected 1s padding, 10-300Hz integrated relative power, de-notch wavelet at 60/120/180±6Hz',
  explicit_limits=dict(primary=('interference','3 channels excluding LOW_SNR only','13-30,30-50'),
    sensitivity=('30-60','rectified','other channels','60Hz flagged exclusion')),
  no_force_emg_synchronization=True,wall_time_s=round(time.time()-t0,2))
 (out/'06_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'07_RESULTS_PRELIMINARY.md').write_text('# V9 Etapa 1 - Morlet bands and MDF compression\n\n'
  +f'Automatic checks: PASS. Participants: {summary["n_participants"]}.\n\n'
  +'This stage analyzes individual EMG recordings; no force/EMG clock synchronization is assumed.\n\n'
  +'Primary unit of inference: participant (mean of eligible 3 channels), 2 disjoint 3-s windows.\n'
  +'The 12 normalized-time trajectories use overlapping windows and remain exploratory.\n\n'
  +dfc.to_markdown(index=False,floatfmt='.4f')+'\n\n'
  +'IMPORTANT: Percent residual is a model mismatch (p.p.), not a neural-drive estimate or a percent contribution of a physiological mechanism.\n'
  +'The rectified outputs are a method sensitivity, not a separate main story.\n',encoding='utf-8')
 (out/'PROCESSING_IN_PROGRESS.txt').unlink(missing_ok=True)
 print(json.dumps(summary,ensure_ascii=False,indent=2));return summary


def test_synthetic():
 fs=1000.;t=np.arange(3000)/fs
 x=np.sin(2*np.pi*20*t)+.12*np.sin(2*np.pi*82*t)
 q=cwt_mean_power(x,fs)
 peak=float(GRID[np.argmax(q)])
 assert abs(peak-20)<5,('Morlet peak',peak)
 d=denotch_spectrum(q)
 b=band_values(q,d)
 assert b['beta_13_30_rel_pct']>b['low_gamma_30_50_rel_pct']
 # Predicted pure frequency compression must relocate energy to lower frequencies.
 f=np.linspace(10,450,400)
 p=np.exp(-.5*((f-100)/15)**2)
 r=.7;pc=compressed_power(p,r,grid=f)
 assert f[np.argmax(pc)]<f[np.argmax(p)]-15
 assert np.allclose(compressed_power(d,1),d)
 print('TEST_SYNTHETIC_PASS: 20 Hz Morlet peak, band classification, downshift and unity rescale')


if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--raw',default=str(RAW_DEFAULT))
 ap.add_argument('--out',default=None)
 ap.add_argument('--demo',default=None)
 ap.add_argument('--test-synthetic',action='store_true')
 a=ap.parse_args()
 if a.test_synthetic:test_synthetic()
 else:
  if not a.out:a.out=str(Path(a.raw).parent/('Paper1_V9_Etapa1_TESTE_ABC' if a.demo else 'Paper1_V9_Etapa1_Resultados'))
  run(a)
