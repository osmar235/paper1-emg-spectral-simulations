#!/usr/bin/env python3
"""Paper1 V7f: paired MU pool / separate RNG streams / best-frequency-scaling controls.

Interpretation: simulation experiments; no human raw EMG or clinical calibration.
V7e source remains frozen. The local copy of tf_emg_sim adds a stream-seeding
hook, without modifying its original simulation kernel for default behavior.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import tf_emg_sim as SIM
import spectral_tools as ST

MASTER_SEED = 20261108
REFERENCE_SHA = 'eebd2e46316afd08558ddf33114e723200c6613a43a5dab431c5254f323cd46d'
TV_MIN = 0.05
SCALING_GRID = np.geomspace(0.50,1.50,121)
STREAMS = ('drive','mu','shape','mun','fib','emg')
MODE_STREAMS = {
    'recording_noise_only': ('mun','emg'),
    'discharge_only': ('drive','mu'),
    'discharge_plus_noise': ('drive','mu','mun','emg'),
    'MUAP_geometry_only': ('shape','fib'),
    'fully_independent': STREAMS,
}


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def normalized(p):
    a=np.maximum(np.asarray(p,float),0)
    s=np.trapezoid(a,ST.FG)
    if not np.isfinite(s) or s<=0:raise ValueError('non-positive PSD area')
    return a/s


def rescaled_pdf(base,ratio):
    if ratio<=0 or not np.isfinite(ratio):raise ValueError('invalid ratio')
    # PDF Jacobian; renormalize only within the fixed observed 10--300 Hz band.
    b=np.interp(ST.FG/ratio,ST.FG,base,left=0,right=0)/ratio
    return normalized(b)


def tv(a,b):
    return float(.5*np.trapezoid(np.abs(a-b),ST.FG))


def compare(base,other,mdf0,mdft,with_oracle=True):
    raw=tv(base,other)
    ratio=float(mdft/mdf0)
    after=tv(other,rescaled_pdf(base,ratio))
    result=dict(MDF_ratio=ratio,TV_total=raw,TV_residual_MDF=after,
        TV_removed_MDF=raw-after,
        MDF_model_worse=int(after>raw+1e-10),
        fraction_removed_MDF=(1-after/raw if raw>=TV_MIN else np.nan))
    if with_oracle:
        # Best post hoc frequency-only scaling. Optimistic IN-SAMPLE bound,
        # not independent validation, and not a new physiology parameter.
        losses=np.array([tv(other,rescaled_pdf(base,s)) for s in SCALING_GRID])
        k=int(np.argmin(losses))
        best=float(SCALING_GRID[k]); bestres=float(losses[k]);
        # s=1 is always part of null model, thus oracle must not be worse.
        if raw < bestres:
            best,bestres=1.0,raw
        result.update(best_scale=best,TV_residual_oracle=bestres,
            fraction_removed_oracle=(1-bestres/raw if raw>=TV_MIN else np.nan),
            TV_removed_oracle=raw-bestres)
    return result


def load_reference():
    p=ROOT/'input_reference'/'EMG_MDF_modelo_por_participante.csv'
    if sha256(p)!=REFERENCE_SHA:
        raise ValueError(f'Human V7d file checksum mismatch at {p}')
    a=pd.read_csv(p,encoding='utf-8-sig')
    if len(a)!=360 or a.force_ID.nunique()!=30 or a.window.nunique()!=12:
        raise ValueError('V7d reference structure unexpectedly changed')
    last=a[a.window==11]
    return float(last.r_MDF_observed.median()), last


def stream_overrides(seed:int,mode:str,rep:int):
    """Vary designated RNG streams, leaving remaining streams EXACTLY fixed."""
    assert mode in MODE_STREAMS
    # deterministic; each (simulation, mode, replication, stream) unique.
    salt=list(MODE_STREAMS).index(mode)+1
    return {stream: int(seed+10000000*salt+100000*rep+1000*(STREAMS.index(stream)+1))
            for stream in MODE_STREAMS[mode]}


def simulate(args,seed,cv_scale,override=None,return_spikes=False):
    x,info=SIM.simulate_tf_segment(duration_s=3.,seed=seed,fs_out=1000.,
          n_mu=args.n_mu,dt=args.dt,n_fibers=args.n_fibers,
          burn_in_s=args.burn_in,dur_rank_link=.9,
          cv_scale=float(cv_scale),cv_gradient=0.,
          stream_seed_overrides=override,return_spikes=return_spikes)
    p=normalized(ST.denotch(ST.psd(x)))
    return x,p,float(ST.mdf(p)),info


def welch_bootstrap(signal,rng,reps=12):
    """Welch-frame resampling conditional on ONE fixed realized time series.

    Adjacent frames overlap and therefore this is a descriptive numerical
    sensitivity control, not a formal CI nor 'pure physiological noise'.
    """
    x=ST.prep(signal)
    freq,_,px=spectrogram(x,fs=1000.,window='hann',nperseg=512,noverlap=256,
                        detrend='constant',scaling='density',mode='psd')
    mask=(freq>=ST.FMIN)&(freq<=ST.FMAX)
    frames=px[mask,:]
    if len(freq[mask])!=len(ST.FG) or np.max(abs(freq[mask]-ST.FG))>1e-8:
        raise ValueError('Incompatible Welch frequency grid')
    # Mainline ST.psd and arithmetic mean of frame PSDs should be identical.
    main=normalized(ST.denotch(ST.psd(signal)))
    same=normalized(ST.denotch(frames.mean(axis=1)))
    if tv(main,same)>1e-8:
        raise ValueError('PSD bootstrap frame estimator differs from V7e Welch')
    outputs=[]
    for b in range(reps):
        p0=normalized(ST.denotch(np.mean(frames[:,rng.integers(0,frames.shape[1],frames.shape[1])],axis=1)))
        p1=normalized(ST.denotch(np.mean(frames[:,rng.integers(0,frames.shape[1],frames.shape[1])],axis=1)))
        outputs.append((b+1,p0,p1))
    return outputs




def check_against_original_v7e(s,args):
    # Strict regression test of original V7e slowing-only experiment; V7f
    # adds stream controls but must not silently change conduction-only data.
    if not (args.subjects==32 and args.stages==12 and args.max_cv_scale==1.6 and
            args.n_mu==72 and args.n_fibers==346 and args.dt==.0002 and args.burn_in==1.):
        return  # custom run, not directly comparable to frozen V7e.
    source=ROOT/'input_reference'/'simulated_slowing_only_V7e.csv'
    old=pd.read_csv(source)
    j=s.merge(old,on=['sim_ID','stage'],how='outer',validate='one_to_one',suffixes=('_new','_V7e'),indicator=True)
    if len(j)!=384 or not j._merge.eq('both').all():
        raise AssertionError('V7e frozen reference participant/stage mismatch')
    for col in ['MDF_ratio','TV_total','TV_residual_MDF']:
        err=np.abs(j[col+'_new'].to_numpy()-j[col+'_V7e'].to_numpy())
        if np.nanmax(err)>1e-7: raise AssertionError(f'V7e regression failed for {col}: {np.nanmax(err)}')
    print('V7e conduction-only regression: PASS (384/384 stages)',flush=True)

def run(args):
    SIM.load_simulator(str(ROOT/'simulator'))
    target,human_last=load_reference()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    levels=np.linspace(1.,args.max_cv_scale,args.stages)
    slowing=[];stochastic=[];welch=[];verification=[];matched=[]
    for i in range(args.subjects):
        seed=MASTER_SEED+2000*i
        ident=f'SIM{i+1:03d}'
        base_x,base,mdf0,base_info=simulate(args,seed,1.,return_spikes=True)
        sp0=base_info.pop('spikes')
        for k,s in enumerate(levels):
            if k==0:post,p,mdft,info=base_x,base,mdf0,base_info
            else:post,p,mdft,info=simulate(args,seed,s,return_spikes=True)
            if k>0:
                if not np.array_equal(sp0,info.pop('spikes')):
                    raise AssertionError('Slowing-only paired spikes changed!')
            if int(info['n_active']) != int(base_info['n_active']):
                raise AssertionError('Slowing-only changed active motor units')
            slowing.append(dict(sim_ID=ident,stage=k,cv_scale=float(s),
                  simulation_stage_position=k/(args.stages-1),
                  active_units=int(base_info['n_active']),
                  MDF_initial_Hz=mdf0,**compare(base,p,mdf0,mdft)))
        # Stationary, repeated conditions with the SAME physical MUAP population
        # unless the mode explicitly changes it.
        for mode in MODE_STREAMS:
            for rep in range(1,args.null_reps+1):
                override=stream_overrides(seed,mode,rep)
                xx,pp,mm,info=simulate(args,seed,1.,override=override,return_spikes=True)
                sn=info.pop('spikes')
                expected_same=mode in ('recording_noise_only','MUAP_geometry_only')
                same_spikes=bool(np.array_equal(sn,sp0))
                if same_spikes!=expected_same:
                    # The source simulator LIF is fully determined by drive and mu.
                    raise AssertionError(f'RNG isolation mismatch for {mode}, {ident}')
                stochastic.append(dict(sim_ID=ident,mode=mode,rep=rep,
                   same_spikes=same_spikes,pool_MUAPs_fixed=(mode!='MUAP_geometry_only' and mode!='fully_independent'),
                   active_units=int(info['n_active']),
                   active_units_changed=int(info['n_active']!=base_info['n_active']),
                   MDF_initial_Hz=mdf0,**compare(base,pp,mdf0,mm)))
        welch_seed=np.random.default_rng(seed+880000)
        for rep,p0,p1 in welch_bootstrap(base_x,welch_seed,reps=args.welch_reps):
            m0=ST.mdf(p0);m1=ST.mdf(p1)
            welch.append(dict(sim_ID=ident,mode='welch_frame_resampling',rep=rep,
                     MDF_initial_Hz=m0,**compare(p0,p1,m0,m1)))
        sl=pd.DataFrame([r for r in slowing if r['sim_ID']==ident])
        chosen=sl.iloc[int(np.argmin(np.abs(np.log(sl.MDF_ratio.to_numpy()/target))))]
        matched.append(dict(sim_ID=ident,target_human_MDF_ratio=target,
               match_stage=int(chosen.stage),match_cv_scale=float(chosen.cv_scale),
               match_MDF_ratio=float(chosen.MDF_ratio),
               match_abs_ratio_error=abs(float(chosen.MDF_ratio)-target),
               match_TV_total=chosen.TV_total,
               match_TV_residual_MDF=chosen.TV_residual_MDF,
               match_TV_residual_oracle=chosen.TV_residual_oracle,
               match_fraction_MDF=chosen.fraction_removed_MDF,
               match_fraction_oracle=chosen.fraction_removed_oracle,
               MDF_predicted_scale=chosen.MDF_ratio,
               oracle_best_scale=chosen.best_scale))
        print(f'[{i+1:02d}/{args.subjects}] {ident}: {args.stages} conduction states, '
              f'{len(MODE_STREAMS)*args.null_reps} stationary controls and '
              f'{args.welch_reps} Welch resamples',flush=True)
    s=pd.DataFrame(slowing);p=pd.DataFrame(stochastic);w=pd.DataFrame(welch);m=pd.DataFrame(matched)
    if not args.quick:
        check_against_original_v7e(s,args)
    for filename,data in [('slowing_paired_full.csv',s),('stationary_controls_by_stream.csv',p),
         ('welch_fixed_signal_resampling.csv',w),('MDF_matched_by_simulation.csv',m)]:
        data.to_csv(out/filename,index=False,float_format='%.10g')
    make_group_summary(s,p,w,m,target,human_last,out)
    write_report(s,p,w,m,target,human_last,args,out)
    primary=(not args.quick and args.subjects==32 and args.stages==12 and
             args.null_reps==2 and args.welch_reps==12 and args.max_cv_scale==1.6 and
             args.n_mu==72 and args.n_fibers==346 and args.dt==.0002 and args.burn_in==1.)
    provenance=dict(status=('QUICK_QA_NOT_SCIENCE' if args.quick else
                          'FULL_V7f_EXPLORATORY' if primary else 'CUSTOM_NOT_PRIMARY'),
        input_human_SHA256=REFERENCE_SHA,seed=MASTER_SEED,
        parameters=dict(subjects=args.subjects,stages=args.stages,max_cv_scale=args.max_cv_scale,
         n_mu=args.n_mu,n_fibers=args.n_fibers,dt=args.dt,burn_in=args.burn_in,
         null_reps=args.null_reps,welch_reps=args.welch_reps),
        source_SHA256={str(f.relative_to(ROOT)):sha256(f) for f in
         [ROOT/'analise_v7f.py',ROOT/'source'/'tf_emg_sim.py',ROOT/'source'/'spectral_tools.py',
         ROOT/'simulator'/'functions.py']},
        outputs_SHA256={f.name:sha256(f) for f in sorted(out.glob('*')) if f.is_file()
                        and f.name not in ('VERIFICACAO_FINAL_V7f.md','PROVENANCE_V7f.json')},
        V7e_frozen_csv_SHA256=sha256(ROOT/'input_reference'/'simulated_slowing_only_V7e.csv'))
    (out/'PROVENANCE_V7f.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nFINALIZADO:',out,flush=True)


def make_group_summary(s,p,w,m,target,human_last,out):
    st=s.groupby('stage').agg(n_sim=('sim_ID','nunique'),
         cv_scale=('cv_scale','first'),MDF_ratio_median=('MDF_ratio','median'),
         TV_total_median=('TV_total','median'),TV_residual_MDF_median=('TV_residual_MDF','median'),
         TV_residual_oracle_median=('TV_residual_oracle','median'),
         frac_MDF_median=('fraction_removed_MDF','median'),
         frac_oracle_median=('fraction_removed_oracle','median')).reset_index()
    st.to_csv(out/'slowing_stage_summary.csv',index=False)
    # Independent simulation is unit of replication; within-MU replicates summarized first.
    control=pd.concat([p,w],ignore_index=True,sort=False)
    bysim=control.groupby(['sim_ID','mode'],as_index=False).agg(
         TV_total=('TV_total','median'), TV_residual_MDF=('TV_residual_MDF','median'),
         TV_residual_oracle=('TV_residual_oracle','median'),
         MDF_ratio=('MDF_ratio','median'))
    bysim.to_csv(out/'stationary_subject_medians.csv',index=False)
    modal=bysim.groupby('mode').agg(n_sim=('sim_ID','nunique'),TV_total_median=('TV_total','median'),
         TV_residual_MDF_median=('TV_residual_MDF','median'),
         TV_residual_oracle_median=('TV_residual_oracle','median'),
         MDF_ratio_median=('MDF_ratio','median')).reset_index()
    modal.to_csv(out/'stationary_mode_summary.csv',index=False)
    simmed=pd.Series({c:float(m[c].median()) for c in ['match_MDF_ratio','match_TV_total',
                       'match_TV_residual_MDF','match_TV_residual_oracle','match_fraction_MDF',
                       'match_fraction_oracle','match_abs_ratio_error']})
    humanmed=pd.Series({'human_MDF_ratio':target,
         'human_TV_total':float(human_last.TV_change_total.median()),
         'human_TV_residual_MDF':float(human_last.TV_change_unexplained_MDF_model.median()),
         'human_TV_residual_oracle':float(human_last.TV_residual_best_dilation.median()),
         'human_fraction_MDF':float(human_last.TV_fraction_reduced.median()),
         'human_fraction_oracle':float(human_last.TV_fraction_reduced_best_dilation.median())})
    with (out/'MDF_matched_group_summary.json').open('w',encoding='utf-8') as f:
        json.dump({'simulation':simmed.to_dict(),'human_reference':humanmed.to_dict()},f,indent=2)
    draw_figure(st,modal,m,humanmed,out)


def draw_figure(stage,modal,matched,human,out):
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=stage.MDF_ratio_median.values*100
    ax[0,0].plot(x,stage.TV_residual_MDF_median,'o-',label='MDF-prescribed')
    ax[0,0].plot(x,stage.TV_residual_oracle_median,'s--',label='Oracle shift (optimistic)')
    ax[0,0].scatter([human.human_MDF_ratio*100],[human.human_TV_residual_MDF],s=65,marker='D',label='Human MDF residual')
    ax[0,0].invert_xaxis();ax[0,0].set_xlabel('Remaining MDF (%)');ax[0,0].set_ylabel('Residual TV (10-300 Hz)')
    ax[0,0].set_title('A. Conduction-only across MDF change');ax[0,0].legend(fontsize=8,frameon=False)
    a=ax[0,1]
    names=modal['mode'].str.replace('_','\n').to_list()
    a.bar(np.arange(len(modal)),modal.TV_residual_MDF_median.values,alpha=.65)
    a.set_xticks(np.arange(len(modal)),names,fontsize=7)
    a.set_ylabel('Median residual TV');a.set_title('B. Stationary, distinct RNG controls')
    a=ax[1,0]
    a.plot(x,stage.frac_MDF_median*100,'o-',label='MDF-prescribed')
    a.plot(x,stage.frac_oracle_median*100,'s--',label='Oracle shift')
    a.axhline(0,color='gray',ls=':');a.set_xlabel('Remaining MDF (%)');a.set_ylabel('Median TV reduction (%)')
    a.invert_xaxis();a.set_title('C. Shift model benefit');a.legend(fontsize=8,frameon=False)
    a=ax[1,1]
    a.scatter(matched.match_MDF_ratio*100,matched.match_TV_residual_MDF,alpha=.65,label='MDF-prescribed')
    a.scatter(matched.match_MDF_ratio*100,matched.match_TV_residual_oracle,alpha=.65,marker='x',label='Oracle')
    a.scatter([human.human_MDF_ratio*100],[human.human_TV_residual_MDF],s=80,marker='D',label='Human median')
    a.set_xlabel('Matched simulated MDF (%)');a.set_ylabel('Residual TV');a.set_title('D. Simulation-level MDF matching');a.legend(fontsize=8,frameon=False)
    for a in ax.flat:
        a.grid(alpha=.18);a.spines[['top','right']].set_visible(False)
    fig.suptitle('Paper 1 V7f | separate stochastic controls (simulation, not physiological attribution)',fontsize=12,weight='bold')
    for ext in ['png','pdf']:
        fig.savefig(out/f'Figure_V7f_Stochastic_Controls.{ext}',dpi=170 if ext=='png' else None)
    plt.close(fig)


def write_report(s,p,w,m,target,human,args,out):
    sim_mdf=float(m.match_MDF_ratio.median()); rm=float(m.match_TV_residual_MDF.median());
    oracle=float(m.match_TV_residual_oracle.median())
    stationary=p.groupby('mode').TV_residual_MDF.median().to_dict()
    txt=[
        '# Paper 1 | V7f -- simulation controls with separate stochastic streams','',
        f'- STATUS: {"SOFTWARE QA ONLY" if args.quick else "FULL EXPLORATORY RUN" if (args.subjects==32 and args.stages==12 and args.null_reps==2) else "CUSTOM NOT PRIMARY"}.',
        f'- {args.subjects} independent simulation seeds, {args.stages} conduction levels, {args.null_reps} stationary replicates/mode and {args.welch_reps} Welch-frame resamples.',
        '- Comparison uses the same MU population geometry/MUAP templates unless mode explicitly indicates otherwise.',
        '- Human V7d data are unchanged (30 individuals).','',
        '## Matched by human median MDF decline',
        f'- Human target MDF ratio (final/initial): {target:.4f}.',
        f'- Simulated MDF ratio, selected by closeness, median: {sim_mdf:.4f}.',
        f'- Simulated TV residual, MDF shift: {rm:.4f}; best oracle shift: {oracle:.4f}.',
        f'- Human residual TV, MDF shift: {human.TV_change_unexplained_MDF_model.median():.4f}; best frequency shift: {human.TV_residual_best_dilation.median():.4f}.',
        '- Oracle shift is optimized against the SAME final PSD: optimistic bound, NOT independently validated.','',
        '## Stationary controls, all cv_scale=1',
    ]
    txt += [f'- {mode}: median residual TV = {value:.4f}' for mode,value in sorted(stationary.items())]
    txt += [f'- Conditional Welch frame-resampling (fixed 3-s signal): median residual TV = {w.TV_residual_MDF.median():.4f}.','',
        '## Controls and limitations','',
        '1. recording_noise_only: same spikes and MUAP geometry; altered per-unit and recording-noise RNG. Simulated SNR scales with waveform standard deviations.',
        '2. discharge_only: MUAP geometry and noise RNG fixed, random synaptic drive/spikes changed; noise amplitude can still change because signal standard deviations vary.',
        '3. discharge_plus_noise: MUAP geometry fixed; both spikes and noise RNG vary.',
        '4. MUAP_geometry_only: same spikes and noise RNG but randomly differing MUAP duration population and fiber projections.',
        '5. fully_independent: changes all six streams (V7e-like independent population control).',
        '6. welch_frame_resampling: resamples overlapping Welch periodogram frames of a fixed signal. It is NOT an independent sample or formal estimator-only variance benchmark.',
        '7. Conduction-only pair holds spikes and RNG streams identical; only cv_scale and derived MUAP duration/inter-electrode delay change.',
        '8. The MDF-prescribed shift is a fixed hypothesis; oracle shift is a descriptive in-sample upper bound on agreement.',
        '9. All PSDs use 10-300 Hz, notch interpolation, and spectral TV. A residual does not identify lactate, acidosis, recruitment, synaptic drive, or clinical pathology.',
        '10. Human and simulated EMG magnitude and noise are not calibrated; no causal subtraction of stationary TV from human TV is justified.',
        '11. Simulation stages are independently synthesized stationary segments, not longitudinal fatigue recordings.',
        '', '## Files','- slowing_paired_full.csv; stationary_controls_by_stream.csv; welch_fixed_signal_resampling.csv; '
        'MDF_matched_by_simulation.csv; slowing_stage_summary.csv; stationary_subject_medians.csv; '
        'stationary_mode_summary.csv; MDF_matched_group_summary.json; Figure_V7f_Stochastic_Controls.png/pdf; PROVENANCE_V7f.json.','']
    (out/'RESULTADOS_V7f.md').write_text('\n'.join(txt),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick',action='store_true',help='Tiny software QA, not scientific results')
    ap.add_argument('--subjects',type=int,default=32)
    ap.add_argument('--stages',type=int,default=12)
    ap.add_argument('--max-cv-scale',type=float,default=1.6)
    ap.add_argument('--null-reps',type=int,default=2)
    ap.add_argument('--welch-reps',type=int,default=12)
    ap.add_argument('--n-mu',type=int,default=72)
    ap.add_argument('--n-fibers',type=int,default=346)
    ap.add_argument('--dt',type=float,default=.0002)
    ap.add_argument('--burn-in',type=float,default=1.0)
    ap.add_argument('--out',type=Path,default=None)
    a=ap.parse_args()
    if a.quick:
        a.subjects=2;a.stages=3;a.n_mu=12;a.n_fibers=64;a.dt=.0005;a.burn_in=.5;a.null_reps=1;a.welch_reps=3
    if not (a.subjects>=1 and a.stages>=3 and a.max_cv_scale>1 and a.null_reps>=1 and a.welch_reps>=2):
        ap.error('invalid simulation parameters')
    if a.out is None:
        a.out=ROOT.parent/('Paper1_V7f_QA_APENAS_TESTE' if a.quick else 'Paper1_V7f_RESULTADOS_COMPLETOS')
    run(a)

if __name__=='__main__':main()
