#!/usr/bin/env python3
"""V9 Stage 4: bilateral motor-unit simulation with known shared oscillatory input.

Paired scenarios at FIXED seeds, model and analysis code. This is a positive-control
study, NOT a quantitative fit to any human subject or a specific cortical source.
"""
from __future__ import annotations
import argparse,sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'simulator'))
sys.path.insert(0,str(ROOT/'modules'))
import functions as f
from coupling_v9_stage2 import cwt_block,metrics
from frozen_stage1_reader import preprocess,welch_mdf
BANDS=('beta_13_30','low_gamma_30_50','piper_30_60_sensitivity')
MEASURES=('cross_wavelet_abs_au','cross_wavelet_rel_pct','coh_welch_mean','wavelet_coh_median')
MODES=('stationary_independent','slowing_only','slowing_rms_matched',
       'shared_input_increase','shared_input_rms_matched',
       'slowing_plus_shared_input','sensor_crosstalk_10pct','amplitude_gain_40pct')
SEED0=2026110904

def digest(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def generate(seed,params,slow=False,info=False):
 config=dict(seed=int(seed),motorunits_max=int(params.nmu),n_fibers=int(params.nfibers),
     dt=float(params.dt),T_end=4.0,burn_in_s=0.8,
     FreqInputTotal=100,SNR=40,SNRE=40,SNRF=8,
     const_current=10.0,common_mod_cortical=1.0,common_mod_subcortical=0.0,
     common_drive_target_rms=float(params.shared),cortical_high_hz=50,
     freq_left=0.0,freq_right=0.0,phase_left=np.pi/2,phase_right=np.pi/2,
     intent_drift='none',Intent_variability=0.0,
     cross_talk_R2L=0.,cross_talk_L2R=0.,
     muap_dur_s=0.024 if slow else 0.018,
     return_spikes=True,progress=False)
 left,right,fs,aux=f.generate_modulated_EMG_physiological_upgraded(**config)
 tau=1.6 if slow else 1.2
 left=f.simulate_bipolar_emg_spatial(left,fs,tau_ms=tau,lp_hz=180)
 right=f.simulate_bipolar_emg_spatial(right,fs,tau_ms=tau,lp_hz=180)
 left=f.resample_to_fs(left,fs,1000.)
 right=f.resample_to_fs(right,fs,1000.)
 if len(left)!=4000 or len(right)!=4000:raise ValueError('Output length not 4 s at 1000 Hz')
 # Same filter used for the observed interference EMG. Filter full segment then crop.
 left=preprocess(left,1000.)[500:3500]
 right=preprocess(right,1000.)[500:3500]
 if not (np.std(left)>0 and np.std(right)>0):raise ValueError('Zero EMG variance')
 return left,right,aux

def match_rms(x,reference):
 return x*(np.std(reference)/max(np.std(x),1e-12))

def scored(x,y):
 a=metrics(x,y,cwt_block(x),cwt_block(y))
 mdfl=float(welch_mdf(x,1000.));mdfr=float(welch_mdf(y,1000.))
 return a,mdfl,mdfr

def main(args):
 dest=Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
 (dest/'RUNNING_NOT_FINISHED.txt').write_text('INCOMPLETE: remove only after successful checks.\n')
 out=[];diag=[]
 for i in range(args.subjects):
  seed=SEED0+2003*i
  config=argparse.Namespace(nmu=args.nmu,nfibers=args.nfibers,dt=args.dt,shared=1.5)
  x0,y0,aux0=generate(seed,config)
  bm,mdfl0,mdfr0=scored(x0,y0)
  outcomes={'baseline':(x0,y0,bm,mdfl0,mdfr0)}
  xstat,ystat,astat=generate(seed+999971,config)
  outcomes['stationary_independent']=(xstat,ystat,*scored(xstat,ystat))
  xs,ys,aslow=generate(seed,config,slow=True)
  assert np.array_equal(aux0['spikes_L'],aslow['spikes_L']) and np.array_equal(aux0['spikes_R'],aslow['spikes_R']), 'Slowing altered spikes'
  outcomes['slowing_only']=(xs,ys,*scored(xs,ys))
  xsm,ysm=match_rms(xs,x0),match_rms(ys,y0)
  outcomes['slowing_rms_matched']=(xsm,ysm,*scored(xsm,ysm))
  config.shared=6.0
  xc,yc,ac=generate(seed,config)
  outcomes['shared_input_increase']=(xc,yc,*scored(xc,yc))
  xcm,ycm=match_rms(xc,x0),match_rms(yc,y0)
  outcomes['shared_input_rms_matched']=(xcm,ycm,*scored(xcm,ycm))
  xcs,ycs,acs=generate(seed,config,slow=True)
  assert np.array_equal(ac['spikes_L'],acs['spikes_L']) and np.array_equal(ac['spikes_R'],acs['spikes_R']), 'Combined slowing altered spikes'
  outcomes['slowing_plus_shared_input']=(xcs,ycs,*scored(xcs,ycs))
  outcomes['sensor_crosstalk_10pct']=(x0+.10*y0,y0+.10*x0,*scored(x0+.10*y0,y0+.10*x0))
  outcomes['amplitude_gain_40pct']=(1.4*x0,1.4*y0,*scored(1.4*x0,1.4*y0))
  for mode,(_,__,sc,mdfl,mdfr) in outcomes.items():
   for r in sc:
    out.append(dict(sim_id='SIM%03d'%(i+1),seed=seed,mode=mode,band=r['band'],
      **{v:float(r[v]) for v in MEASURES},mdf_left_hz=mdfl,mdf_right_hz=mdfr))
  diag.append({'sim_id':'SIM%03d'%(i+1),'seed':seed,'baseline_spikes_match_slow':True,
    'shared_spikes_match_combined':True,'baseline_mdf_left':mdfl0,'baseline_mdf_right':mdfr0,
    'common_drive_baseline_rms':1.5,'common_drive_increased_rms':6.0})
  print(f'{i+1}/{args.subjects}: simulations and 7 conditions completed',flush=True)
 d=pd.DataFrame(out);pd.DataFrame(diag).to_csv(dest/'01_MECHANISM_QA.csv',index=False)
 d.to_csv(dest/'02_ALL_METRICS.csv',index=False,float_format='%.10g')
 base=d[d['mode']=='baseline'].set_index(['sim_id','band'])
 changed=[]
 for mode in MODES:
  g=d[d['mode']==mode].set_index(['sim_id','band'])
  if not g.index.equals(base.index):g=g.reindex(base.index)
  for idx in base.index:
   before=base.loc[idx];after=g.loc[idx]
   changed.append(dict(sim_id=idx[0],band=idx[1],mode=mode,
     cross_abs_ratio=float(after.cross_wavelet_abs_au/max(before.cross_wavelet_abs_au,1e-12)),
     log_cross_abs_ratio=float(np.log(max(after.cross_wavelet_abs_au,1e-12)/max(before.cross_wavelet_abs_au,1e-12))),
     delta_cross_rel_pct=float(after.cross_wavelet_rel_pct-before.cross_wavelet_rel_pct),
     delta_welch=float(after.coh_welch_mean-before.coh_welch_mean),
     delta_wtc=float(after.wavelet_coh_median-before.wavelet_coh_median),
     avg_mdf_ratio=float(.5*(after.mdf_left_hz/max(before.mdf_left_hz,1e-12)+after.mdf_right_hz/max(before.mdf_right_hz,1e-12)))))
 cd=pd.DataFrame(changed)
 cd.to_csv(dest/'03_PAIRED_EFFECTS.csv',index=False,float_format='%.10g')
 summaries=[]
 for (mode,band),g in cd.groupby(['mode','band']):
  record=dict(mode=mode,band=band,n_realizations=len(g))
  for m in ('log_cross_abs_ratio','delta_cross_rel_pct','delta_welch','delta_wtc','avg_mdf_ratio'):
   val=g[m].to_numpy(float);record[f'median_{m}']=float(np.median(val));record[f'n_positive_{m}']=int((val>0).sum())
   if m!='avg_mdf_ratio':record[f'p_wilcoxon_exploratory_{m}']=float(wilcoxon(val).pvalue) if len(val)>=6 and not np.allclose(val,0,atol=1e-12) else np.nan
  record['median_cross_abs_gain_ratio']=float(np.exp(record['median_log_cross_abs_ratio']))
  summaries.append(record)
 sm=pd.DataFrame(summaries);sm.to_csv(dest/'04_GROUP_SUMMARY.csv',index=False,float_format='%.10g')
 fig,axs=plt.subplots(2,2,figsize=(11.5,8),layout='constrained')
 chosen=('beta_13_30','low_gamma_30_50')
 for j,band in enumerate(chosen):
  plot_modes=('stationary_independent','slowing_only','slowing_rms_matched','shared_input_increase','shared_input_rms_matched','sensor_crosstalk_10pct','amplitude_gain_40pct')
  xx=sm[sm.band==band].set_index('mode').loc[list(plot_modes)]
  axs[0,j].bar(np.arange(len(xx)),xx.median_cross_abs_gain_ratio)
  axs[0,j].axhline(1.,color='black',ls='--',lw=1)
  axs[0,j].set(ylabel='Median cross-wavelet abs. ratio vs baseline',title=band,xticks=np.arange(len(xx)),xticklabels=['Stationary','Slowing','Slowing + RMS match','Common input','Common + RMS match','Crosstalk','Amplitude'])
  axs[0,j].tick_params(axis='x',rotation=35)
  axs[1,j].plot(np.arange(len(xx)),xx.median_delta_wtc,'o-',label='Wavelet coherence')
  axs[1,j].plot(np.arange(len(xx)),xx.median_delta_welch,'s-',label='Welch coherence')
  axs[1,j].axhline(0.,color='black',ls='--',lw=1)
  axs[1,j].set(ylabel='Median change in coherence',xticks=np.arange(len(xx)),xticklabels=['Stationary','Slowing','Slowing + RMS match','Common input','Common + RMS match','Crosstalk','Amplitude'])
  axs[1,j].tick_params(axis='x',rotation=35)
 axs[1,0].legend(frameon=False)
 fig.suptitle('SIMULATION ONLY | known mechanisms, NOT calibrated to human data')
 fig.savefig(dest/'Figure_Simulated_Mechanism_Controls.png',dpi=155)
 plt.close(fig)
 expected=args.subjects*(1+len(MODES))*len(BANDS)
 if len(d)!=expected or len(cd)!=args.subjects*len(MODES)*len(BANDS):raise RuntimeError('Missing simulated modes or bands')
 if d[list(MEASURES)].isna().any().any():raise RuntimeError('Invalid coupling metric')
 amp=cd[cd['mode']=='amplitude_gain_40pct']
 if not np.allclose(amp.cross_abs_ratio,1.4**2,rtol=3e-6,atol=3e-6):
  raise RuntimeError('Amplitude-only cross-wavelet scaling control failed')
 if not np.allclose(amp[['delta_wtc','delta_welch']].to_numpy(),0,atol=2e-5):
  raise RuntimeError('Amplitude-only coherence invariance control failed')
 verification=dict(status='PASS' if args.subjects==32 else 'QA_ONLY',n_simulations=args.subjects,
     n_channels_per_simulation=2,n_muscles_per_side=args.nmu,n_fibers=args.nfibers,
     n_modes_with_baseline=1+len(MODES),n_metric_rows=len(d),n_effect_rows=len(cd),dt=args.dt,
     shared_drive_rms_baseline=1.5,shared_drive_rms_high=6.0,
     no_human_raw_or_statistical_tuning=True,
     simulator_sha256=digest(ROOT/'simulator'/'functions.py'),
     interpretation='Known shared oscillator is model input, not evidence of cortical origin')
 (dest/'05_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
 lines=[]
 for ff in sorted(dest.iterdir()):
  if ff.is_file() and ff.name not in ('SHA256_RESULTS.txt','RUNNING_NOT_FINISHED.txt'):
   lines.append(digest(ff)+'  '+ff.name+'\n')
 (dest/'SHA256_RESULTS.txt').write_text(''.join(lines))
 (dest/'RUNNING_NOT_FINISHED.txt').unlink()
 print(json.dumps(verification,indent=2),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--subjects',type=int,default=32)
 ap.add_argument('--nmu',type=int,default=48)
 ap.add_argument('--nfibers',type=int,default=64)
 ap.add_argument('--dt',type=float,default=0.0005)
 ap.add_argument('--out',type=Path,default=ROOT.parent/'Paper1_V9_Etapa4_Resultados')
 args=ap.parse_args()
 main(args)
