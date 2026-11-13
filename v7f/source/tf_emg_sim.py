"""
tf_emg_sim.py - simulation of surface EMG during a sustained high-force isometric
contraction to task failure, built on the EMG_Simulator_WaveletCoherence toolbox
(Pinto Neto et al., Comput Methods Programs Biomed 2026; 109657,
https://doi.org/10.1016/j.cmpb.2026.109657; software https://doi.org/10.5281/zenodo.19703513).

The toolbox (functions.py) is NOT copied here: download it (see README.md) and
point to it with --simulator / the EMGSIM_PATH environment variable. This module
re-uses its building blocks (LIF motor-neuron pool, MUAP templates, fibre
dispersion, bipolar projection, noise) and adds what a single-muscle fatigue
study needs:

  * one motor-unit pool at a constant (isometric) force level with a tonic
    excitatory drive, so that units discharge regularly (mean-driven firing);
  * independent synaptic noise for every motor unit, on top of the drive shared
    by the pool, so that discharge variability and synchrony are realistic;
  * separate common-drive components (alpha 5-13, beta 13-30, gamma 30-60 Hz);
  * the manipulations used to emulate fatigue in the END state:
      cv_scale        conduction-velocity slowing: MUAP duration and bipolar
                      delay x cv_scale (1 = none; 1.4 = 40 % longer MUAPs)
      cv_gradient     0 = uniform slowing; > 0 = larger (high-threshold) units
                      slow more (mean slowing unchanged)
      tonic           tonic excitatory drive (sets discharge rates / recruitment)
      beta_sd, alpha_sd, gamma_sd   common-drive amplitudes
      mu_noise_sd     independent per-unit noise (lower = more synchrony)
      derecruit_frac  fraction of highest-threshold units that stop firing
      pool_sd         broadband (0-100 Hz) input shared by the whole pool
                      (source of short-term synchrony and force fluctuations)
  * optionally, a force proxy (sum of motor-unit twitches) and its detrended CV

Main function: simulate_tf_segment(duration_s, seed, **params)
"""
import os
import sys
import numpy as np

# ----------------------------------------------------------------------------- simulator
F = None


def load_simulator(path=None, min_version="2.0.0"):
    """Import functions.py from the downloaded EMG_Simulator_WaveletCoherence folder."""
    global F
    path = path or os.environ.get("EMGSIM_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                  "EMG_Simulator_WaveletCoherence")
    path = os.path.abspath(path)
    if not os.path.isfile(os.path.join(path, "functions.py")):
        raise FileNotFoundError(
            f"functions.py not found in {path}.\nDownload the simulator first (README.md, step 1) or pass "
            f"--simulator <folder containing functions.py>.")
    if path not in sys.path:
        sys.path.insert(0, path)
    import functions as _F
    v = tuple(int(x) for x in _F.__version__.split(".")[:3])
    if v < tuple(int(x) for x in min_version.split(".")):
        raise RuntimeError(f"Simulator version {_F.__version__} found; version >= {min_version} is required.")
    F = _F
    return F


# ----------------------------------------------------------------------------- parameters
BASELINE = dict(
    n_mu=72, dt=0.0002, burn_in_s=1.0,
    # motor-neuron pool (same LIF model and parameters as the toolbox)
    Vth_min=-0.055, Vth_max=-0.040, V_reset=-0.080, V_e=-0.075, Rm=10e6, tau_m=10e-3,
    min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0, I_scale=1.5e-9,
    # synaptic input (dimensionless units, x I_scale)
    tonic=3.2,               # tonic excitatory drive
    pool_sd=0.6, pool_fc=100.0,     # broadband drive shared by the pool (low-pass pool_fc Hz)
    alpha_sd=0.25, beta_sd=0.35, gamma_sd=0.0,   # common oscillatory drive (5-13, 13-30, 30-60 Hz)
    mu_noise_sd=1.6, mu_noise_fc=100.0,          # independent per-unit synaptic noise
    # MUAPs and recording
    muap_dur_s=0.018, muap_spread=0.45,  # mean duration; per-unit uniform spread (+-fraction)
    dur_rank_link=0.0,       # 0 = MUAP duration random across units; 1 = shortest MUAPs in the
                             # highest-threshold units (faster fibres), linearly by rank
    gain_x_high=1.0,         # amplitude multiplier of the high-threshold half, ramping 1 -> x with rank
    gain_x=1.0,              # uniform MUAP amplitude multiplier (e.g. potentiation); scales the whole
                             # signal (noise is set relative to it), so spectral shape is unchanged
    gain_low=0.6, gain_high=1.4, n_fibers=346,
    snr_mu_db=8.0, snr_emg_db=40.0,
    bipolar_tau_ms=1.2, bipolar_lp_hz=None,
    # fatigue manipulations (defaults = none)
    cv_scale=1.0, cv_gradient=0.0, derecruit_frac=0.0,
)


def _zrows(x):
    x = np.asarray(x, float)
    return (x - x.mean(-1, keepdims=True))/(x.std(-1, keepdims=True) + 1e-12)


def _band(rng, n, lo, hi, fs):
    x = rng.standard_normal(n)
    x = F.butter_filter_low(x, hi, fs)
    x = F.butter_filter_high(x, lo, fs)
    return _zrows(x)


def _snr_sd(sd, snr_db):
    return 0.0 if sd <= 0 else sd/(10.0**(snr_db/20.0))


# ----------------------------------------------------------------------------- simulator core
def force_from_spikes(spikes, dt, twitch_range=30.0, t_longest_ms=90.0, contraction_time_range=3.0):
    """Force proxy: linear sum of motor-unit twitches (Fuglevand et al. 1993 twitch model).

    Unit i (ordered low -> high threshold) has peak twitch force P_i = exp(ln(twitch_range) i/(n-1))
    and contraction time T_i = t_longest * (1/P_i)^(1/c), c = log_{contraction_time_range}(twitch_range);
    twitch f(t) = P_i t/T_i exp(1 - t/T_i). Summation is linear (no rate-dependent gain), so the
    output is a relative force signal used for its fluctuations (CV), not for absolute force.
    """
    from scipy.signal import fftconvolve
    n_mu, N = spikes.shape
    P = np.exp(np.log(twitch_range)*np.arange(n_mu)/max(n_mu - 1, 1))
    c = np.log(twitch_range)/np.log(contraction_time_range)
    T = t_longest_ms*1e-3*(1.0/P)**(1.0/c)
    t = np.arange(int(round(5*T.max()/dt)) + 1)*dt
    force = np.zeros(N)
    for i in np.flatnonzero(spikes.any(1)):
        tw = P[i]*t/T[i]*np.exp(1.0 - t/T[i])
        force += fftconvolve(spikes[i].astype(float), tw)[:N]
    return force


def force_cv_percent(force, fs, fs_force=100.0):
    """Detrended force CV (%) after resampling to the athletes' force sampling rate (100 Hz)."""
    x = F.resample_to_fs(force, fs, fs_force)
    tt = np.arange(x.size); res = x - np.polyval(np.polyfit(tt, x, 1), tt)
    return float(np.std(res, ddof=1)/np.mean(x)*100)


def simulate_tf_segment(duration_s=3.0, seed=0, fs_out=1000.0, return_spikes=False, return_force=False,
                        stream_seed_overrides=None, **params):
    """Simulate one stationary segment of bipolar surface EMG from one muscle.

    Parameters: any key of BASELINE (see module docstring). The first
    `burn_in_s` seconds are simulated and discarded.
    Returns emg (bipolar, resampled to fs_out) and an info dict
    (discharge rates, ISI CV, active units, effective slowing; with
    return_force=True also the twitch-sum force proxy and its detrended CV).
    """
    if F is None:
        load_simulator()
    p = dict(BASELINE); p.update(params)
    unknown = set(params) - set(BASELINE)
    if unknown:
        raise TypeError(f"unknown parameter(s): {sorted(unknown)}")
    dt = p["dt"]; fs = 1.0/dt; n_mu = int(p["n_mu"])
    n_out = int(round(duration_s/dt)); n_burn = int(round(p["burn_in_s"]/dt)); N = n_out + n_burn
    ss = np.random.SeedSequence(seed)
    # V7f extension: replace only chosen random streams. If None, this exactly
    # retains V7e RNG behavior (six child SeedSequences of the same seed).
    streams = ss.spawn(6)
    if stream_seed_overrides is not None:
        names = {"drive": 0, "mu": 1, "shape": 2, "mun": 3, "fib": 4, "emg": 5}
        invalid = set(stream_seed_overrides) - set(names)
        if invalid:
            raise ValueError(f"Invalid stream_seed_overrides: {sorted(invalid)}")
        for name, override_seed in stream_seed_overrides.items():
            streams[names[name]] = np.random.SeedSequence(int(override_seed))
    s_drive, s_mu, s_shape, s_mun, s_fib, s_emg = streams
    rd = np.random.default_rng(s_drive)

    # --- synaptic input: tonic + pool-shared broadband + common oscillations + per-unit noise
    shared = p["tonic"] + p["pool_sd"]*_zrows(F.butter_filter_low(rd.standard_normal(N), p["pool_fc"], fs))
    for key, (lo, hi) in (("alpha_sd", (5, 13)), ("beta_sd", (13, 30)), ("gamma_sd", (30, 60))):
        if p[key] > 0:
            shared = shared + p[key]*_band(rd, N, lo, hi, fs)
    rmu = np.random.default_rng(s_mu)
    own = p["mu_noise_sd"]*_zrows(F.butter_filter_low(rmu.standard_normal((n_mu, N)), p["mu_noise_fc"], fs)) \
        if p["mu_noise_sd"] > 0 else np.zeros((n_mu, N))
    Im = p["I_scale"]*(shared[None, :] + own)

    # --- motor-neuron pool (continuous LIF with adaptation, toolbox implementation)
    Th = np.linspace(p["Vth_min"], p["Vth_max"], n_mu)
    spikes, _ = F.lif_population(Im, np.arange(n_mu), Th, p["V_reset"], p["V_e"], p["Rm"], p["tau_m"], dt,
                                 adaptation=True, min_isi_ms=p["min_isi_ms"], dVth=p["dVth"],
                                 tau_adapt_ms=p["tau_adapt_ms"])
    n_off = int(round(p["derecruit_frac"]*n_mu))
    if n_off > 0:
        spikes[n_mu - n_off:] = False                      # highest-threshold units stop firing

    # --- MUAPs: per-unit duration, fatigue slowing (uniform or threshold-graded)
    rs = np.random.default_rng(s_shape)
    rank = np.linspace(-1.0, 1.0, n_mu)                    # low -> high threshold
    link = p["dur_rank_link"]                              # 0: duration independent of threshold
    base_dur = p["muap_dur_s"]*(1.0 + p["muap_spread"]*((1.0 - link)*rs.uniform(-1, 1, n_mu) - link*rank))
    k_i = 1.0 + (p["cv_scale"] - 1.0)*(1.0 + p["cv_gradient"]*rank)
    durs = base_dur*k_i
    gains = np.linspace(p["gain_low"], p["gain_high"], n_mu)*(1.0 + (p["gain_x_high"] - 1.0)*np.clip(rank, 0, None))*p["gain_x"]
    rmn = np.random.default_rng(s_mun)
    fib = s_fib.spawn(n_mu)
    emg = np.zeros(N)
    for i in range(n_mu):
        if not spikes[i].any():
            continue
        shape = F.generate_muap_shape_dog(durs[i], fs, mu_factor=i/(n_mu - 1 + 1e-12))
        conv = np.convolve(spikes[i].astype(float), shape, mode="same")*gains[i]
        conv += F.pinkish_noise_fft(N, fs, alpha=1.2, f_lo=10, f_hi=450, rng=rmn)*_snr_sd(conv.std(), p["snr_mu_db"])
        emg += F.simulate_fiber_emg_vectorized(conv, p["n_fibers"], 1, rng=np.random.default_rng(fib[i]))
    emg = emg[n_burn:]
    re = np.random.default_rng(s_emg)
    emg = emg + F.pinkish_noise_fft(emg.size, fs, alpha=1.2, f_lo=10, f_hi=450, rng=re)*_snr_sd(emg.std(), p["snr_emg_db"])

    # --- bipolar recording; the inter-electrode delay scales with the mean slowing of active units
    sp = spikes[:, n_burn:]
    counts = sp.sum(1)
    active = counts > 0
    k_eff = float(np.average(k_i[active], weights=counts[active])) if active.any() else float(p["cv_scale"])
    bip = F.simulate_bipolar_emg_spatial(emg, fs, tau_ms=p["bipolar_tau_ms"]*k_eff, lp_hz=p["bipolar_lp_hz"])
    out = F.resample_to_fs(bip, fs, fs_out)

    rates = counts/duration_s
    cv = []
    for i in np.flatnonzero(counts > 4):
        isi = np.diff(np.flatnonzero(sp[i]))*dt
        cv.append(isi.std()/isi.mean())
    info = dict(fs=fs_out, n_active=int(active.sum()), rate_median=float(np.median(rates[active])) if active.any() else 0.0,
                rate_mean=float(rates[active].mean()) if active.any() else 0.0,
                isi_cv_median=float(np.median(cv)) if cv else np.nan, k_eff=k_eff)
    if return_spikes:
        info["spikes"] = sp
    if return_force:                                       # twitches from burn-in spikes carry over
        frc = force_from_spikes(spikes, dt)[n_burn:]
        info["force"] = F.resample_to_fs(frc, fs, 100.0)
        info["force_cv"] = force_cv_percent(frc, fs)
    return out, info
