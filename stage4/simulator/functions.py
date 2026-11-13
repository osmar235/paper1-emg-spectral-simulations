"""
EMG_Simulator_WaveletCoherence: bilateral surface-EMG simulator and EMG-EMG
wavelet-coherence toolbox.  https://github.com/osmar235/EMG_Simulator_WaveletCoherence

Pinto Neto O, Pinho TOR, Wang Y, Balbinot G, Kennedy DM. A Computational Framework
for EMG Simulation and Coherence-Based Biomarker Analysis on Neural Crosstalk in
Bimanual Coordination. Computer Methods and Programs in Biomedicine (2026) 109657.
https://doi.org/10.1016/j.cmpb.2026.109657

If you use this code (simulation or coherence analysis), please cite the article
above and the archived software (see CITATION.cff / README.md).

Main entry points
-----------------
Simulation
    simulate_bilateral_emg(...)                       -> left, right, fs[, info]
    generate_modulated_EMG_physiological_upgraded(...) (full parameter set)
    simulate_bipolar_emg_spatial(x, fs)               monopolar -> bipolar
Coherence analysis (works on simulated or recorded EMG)
    preprocess_emg_for_coherence(x, fs)               band-pass, detrend, z-score
    compute_wavelet_coherence(x, y, fs)               squared Morlet wavelet coherence
    band_medians(Rsq, freqs, period=, coi=)           band summaries inside the COI
    cross_trial_null(left_trials, right_trials, fs)   matched cross-trial null / p-values
"""

__version__ = "2.0.0"

import itertools
import numpy as np
from fractions import Fraction
from tqdm.auto import tqdm
from scipy.signal import (butter, filtfilt, iirnotch, resample_poly, convolve2d,
                          detrend as scipy_detrend, windows as signal_windows)

try:
    import pycwt as wavelet
except ImportError:  # simulation-only use does not need pycwt
    wavelet = None

DEFAULT_BANDS = ((5, 13), (13, 30), (30, 60), (60, 100))

# =============================================================================
# Basic signal utilities
# =============================================================================

def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def butter_filter_low(x, fc, fs, order=4):
    b, a = butter(order, fc/(fs/2), btype="low")
    return filtfilt(b, a, x)


def butter_filter_high(x, fc, fs, order=4):
    b, a = butter(order, fc/(fs/2), btype="high")
    return filtfilt(b, a, x)


def bandpass_emg(x, fs, low=5, high=499, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, x)


def resample_to_fs(x, fs_in, fs_out, max_den=10000):
    """Polyphase resampling (with anti-alias filtering) from fs_in to fs_out."""
    if np.isclose(fs_in, fs_out, rtol=1e-8, atol=1e-12):
        return np.asarray(x, float)
    frac = Fraction(fs_out/fs_in).limit_denominator(max_den)
    return resample_poly(np.asarray(x, float), frac.numerator, frac.denominator)


def notch_filter(x, fs, f0=60.0, harmonics=3, q=30.0):
    """Zero-phase IIR notch at f0 and its harmonics below Nyquist (power-line removal)."""
    y = np.asarray(x, float)
    for k in range(1, int(harmonics) + 1):
        f = k*float(f0)
        if f >= fs/2:
            break
        b, a = iirnotch(f, q, fs)
        y = filtfilt(b, a, y)
    return y

# =============================================================================
# Optional analysis taper: Matviyenko optimized window
# =============================================================================
# Coefficients of the optimized spectral window of G. Matviyenko, "Optimized local
# trigonometric bases", Appl. Comput. Harmon. Anal. 3 (1996) 301-323, following the
# MATLAB implementation by F. Meyer (2003). matviyenko(K, n) returns a symmetric
# window of 2*n samples; larger K gives steeper transitions and lower side lobes.
_MATVIYENKO_G = {
    2: [1.1002143947640],
    3: [1.1723768006269012949, 0.1855148479250006034],
    4: [1.2031447668472587192, 0.2487749850071917170, 0.0475141123801596348],
    5: [1.2196727213232474166, 0.2853161868567129887, 0.0789155484618257136, 0.0135550276613148530],
    6: [1.2299263341780548351, 0.3091936162560012950, 0.1022547499107077172, 0.0270080848154667718, 0.0040643028367457815],
    7: [1.2368960390161255733, 0.3260454684070418003, 0.1202144284772571658, 0.0392824672307590613, 0.0094646390135188831, 0.0012540398189263445],
    8: [1.2419380540495429195, 0.3385842628287665721, 0.1344330305316184567, 0.0501467090467806012, 0.0153116614831942916, 0.0033448841387432324, 0.0003942039965801714],
    9: [1.2457536263723204704, 0.3482817014907560773, 0.1459560151659402844, 0.0596831230559218041, 0.0211609941444967639, 0.0059658781972807312, 0.0011854380315612793, 0.0001255469017738763],
    10: [1.2487411472878298253, 0.3560069611425606312, 0.1554771240803420597, 0.0680536775001451859, 0.0267991398186155608, 0.0088894805175620360, 0.0023126782210685839, 0.0004203150366958263, 0.0000403732905776057],
    11: [1.2511435101979626018, 0.3623069592485838316, 0.1634729471600794903, 0.0754255125823668245, 0.0321305559573675648, 0.0119592651970663211, 0.0036995960643158676, 0.0008907386396059369, 0.0001489420206098608, 0.0000130803773243655],
    12: [1.2531171572760240244, 0.3675432476836647534, 0.1702809341651916143, 0.0819484046917260704, 0.0371207535134114532, 0.0150712511279723392, 0.0052752483692935777, 0.0015235383929696997, 0.0003408100998412218, 0.0000527237866578197, 0.0000042630197561061],
    13: [1.2547673684254528441, 0.3719644997534442686, 0.1761463604899294465, 0.0877499693755897336, 0.0417667654449872536, 0.0181579310814718085, 0.0069802273692443469, 0.0022969056584170223, 0.0006209177060710873, 0.0001295783756168840, 0.0000186407482263636, 0.0000013960635001692],
}


def matviyenko(K, n):
    """Matviyenko optimized window of exactly 2*n samples (K = 2..13)."""
    if isinstance(K, bool) or int(K) != K or not (2 <= int(K) <= 13):
        raise ValueError('K must be an integer from 2 through 13')
    if isinstance(n, bool) or int(n) != n or int(n) < 1:
        raise ValueError('n must be a positive integer half-window length')
    K, n = int(K), int(n)
    coeff = np.asarray(_MATVIYENKO_G[K], dtype=float)
    x = (np.arange(1, n + 1, dtype=float) / n) - 0.5
    harmonic = np.arange(K - 1, dtype=float) + 0.5
    S = np.sin(np.pi * x[:, None] * harmonic[None, :])
    r = 0.5 * (1.0 + S @ coeff)
    b = np.empty(2 * n, dtype=float)
    b[:n] = r
    b[n:] = r[::-1]
    return b


def matviyenko_window(length, K=8):
    """Whole-record Matviyenko window for an even-length signal (rises over the
    first half of the record and falls over the second half)."""
    if isinstance(length, bool) or int(length) != length or int(length) < 2:
        raise ValueError('length must be an integer >= 2')
    length = int(length)
    if length % 2:
        raise ValueError('matviyenko_window requires an even signal length (2*n)')
    return matviyenko(K, length // 2)

# =============================================================================
# Bipolar (spatial) projection
# =============================================================================

def _frac_delay(x, D, N=21, pad_mode="reflect"):
    x = np.asarray(x, float)
    n = np.arange(N)
    h = np.sinc(n - (N-1)/2 - D) * np.hamming(N)
    h /= h.sum()
    pad = (N-1)//2 + int(np.ceil(abs(D)))
    xpad = np.pad(x, (pad, pad), mode=pad_mode) if pad > 0 else x
    ypad = np.convolve(xpad, h, mode="same")
    return ypad[pad:-pad] if pad > 0 else ypad


def simulate_bipolar_emg_spatial(x, fs, tau_ms=1.2, lp_hz=180, wA=1.0, wB=1.0, fir_len=21):
    """Bipolar surface EMG from a monopolar signal: x(t) - x(t - tau), where tau
    is the inter-electrode propagation delay, followed by an optional low-pass
    (tissue/electrode filtering). Set lp_hz=None to skip the low-pass."""
    D = tau_ms*1e-3*fs
    chA = x
    chB = _frac_delay(x, D, N=fir_len, pad_mode="reflect")
    y = wA*chA - wB*chB
    if lp_hz is not None:
        b, a = butter(4, lp_hz/(fs/2), btype="low")
        y = filtfilt(b, a, y, padtype="odd", padlen=3*(max(len(a), len(b))-1))
    return y

# =============================================================================
# Noise helpers
# =============================================================================

def pinkish_noise_fft(N, fs, alpha=1.0, f_lo=10, f_hi=400, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(N)
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    shape = np.ones_like(freqs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    shape[~mask] = 0.0
    shape[mask] = np.maximum(freqs[mask], 1e-3)**(-alpha/2.0)
    y = np.fft.irfft(W*shape, n=N)
    return zscore(y)


def colored_noise_from_shaper(N, shaper_taps, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    w = rng.standard_normal(N + len(shaper_taps) - 1)
    y = np.convolve(w, shaper_taps, mode='valid')[:N]
    return zscore(y)


def _noise(N, fs, noise_type, rng, shaper_taps=None):
    """Unit-variance noise of the requested colour, drawn from `rng`."""
    if noise_type == 'pink':
        return pinkish_noise_fft(N, fs, alpha=1.2, f_lo=10, f_hi=450, rng=rng)
    if noise_type == 'shaped' and shaper_taps is not None:
        return colored_noise_from_shaper(N, shaper_taps, rng=rng)
    return zscore(rng.standard_normal(N))


def _snr_noise_sd(signal_sd, snr_db):
    """Noise SD giving the requested SNR in dB (20*log10 of the SD ratio)."""
    signal_sd = float(signal_sd)
    if signal_sd <= 0:
        return 0.0
    return signal_sd / (10.0 ** (float(snr_db) / 20.0))


def measured_snr_db(signal, noise):
    s = np.std(np.asarray(signal, float))
    n = np.std(np.asarray(noise, float))
    return np.inf if n == 0 else 20.0*np.log10(s/n)

# =============================================================================
# MUAP templates, force envelope
# =============================================================================

def generate_muap_shape_dog(dur_s, fs, width_scale=1.0, tri_ratio=0.35, mu_factor=0.0):
    """Difference-of-Gaussians MUAP template of duration dur_s (peak-normalised)."""
    L = max(16, int(round(dur_s*fs)))
    t = np.linspace(-0.5, 0.5, L)
    s1 = 0.11*width_scale*(1 - 0.25*mu_factor)
    s2 = 0.20*width_scale*(1 - 0.15*mu_factor)
    g1 = np.exp(-0.5*(t/s1)**2); g2 = np.exp(-0.5*(t/s2)**2)
    dog = g1 - 0.65*g2
    if tri_ratio > 0:
        s3 = 0.28 * width_scale
        g3 = np.exp(-0.5*(t/s3)**2)
        dog = dog + tri_ratio*(g3 - g3.mean())
    dog -= dog.mean()
    dog /= (np.max(np.abs(dog)) + 1e-12)
    return dog.astype(float)


def generate_muap_shape_original(d, dt, tau=0.18):
    """Alternative biphasic MUAP template of duration d."""
    n = max(2, int(round((d/2)/dt)))
    t = np.linspace(0, d/2, n)
    shape1 = 5*np.sin(np.pi*t/(d/2)) * np.exp((1/tau)*((t/(d/2))-1))
    shape2 = -np.flip(shape1)[1:]
    return np.concatenate([shape1, shape2])


def force_modulation_sine(t, f, phase=0):
    """Force envelope in [0, 1]. Use f=0 with phase=pi/2 for a constant envelope."""
    return (np.sin(2*np.pi*f*t + phase) + 1.0)/2.0

# =============================================================================
# Leaky integrate-and-fire motor-neuron pool
# =============================================================================

def lif_population(Im, group, Vth_base, V_reset, V_e, Rm, tau_m, dt,
                   state=None, adaptation=True,
                   min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0):
    """Vectorised LIF motor-neuron pool with after-spike threshold adaptation.

    Parameters
    ----------
    Im : array (n_groups, L)
        Input current of each group (e.g. one row per limb).
    group : int array (n_units,)
        Row of `Im` that drives each unit.
    Vth_base : array (n_units,)
        Static threshold of each unit (V).
    state : dict or None
        {'V', 'Vth', 'since'} returned by a previous call, to continue the
        integration seamlessly. None = fresh start.

    Update rule per sample t:
        Vth <- Vth_base + (Vth - Vth_base) * exp(-dt/tau_adapt)
        if t - t_last < min_isi : V[t+1] = V_reset                 (refractory)
        elif V[t] >= Vth        : spike at t, V[t+1] = V_reset, Vth += dVth
        else                    : V[t+1] = V[t] + dt*(-(V[t]-V_e) + Im[t]*Rm)/tau_m
    With adaptation=False the refractory period and the threshold step are disabled.

    Returns
    -------
    spikes : bool array (n_units, L)
    state  : dict for continuing the integration.
    """
    Im = np.atleast_2d(np.asarray(Im, float))
    group = np.asarray(group, int)
    Vb = np.asarray(Vth_base, float)
    n, L = Vb.size, Im.shape[1]
    if state is None:
        V = np.full(n, float(V_reset)); Vth = Vb.copy(); since = np.full(n, 10**9, dtype=np.int64)
    else:
        V = np.asarray(state['V'], float).copy(); Vth = np.asarray(state['Vth'], float).copy()
        since = np.asarray(state['since'], np.int64).copy()
    if adaptation:
        min_isi = int(round(min_isi_ms/1000.0/dt))
        decay = np.exp(-dt/max(tau_adapt_ms/1000.0, 1e-6))
        step = dVth
    else:
        min_isi, decay, step = 0, 0.0, 0.0
        Vth = Vb.copy()
    last = -since
    k = dt/tau_m
    ImT = np.ascontiguousarray(Im.T)
    spikes = np.zeros((n, L), dtype=bool)
    for t in range(L):
        Vth = Vb + (Vth - Vb)*decay
        refr = (t - last) < min_isi
        fire = (~refr) & (V >= Vth)
        Vn = V + k*(-(V - V_e) + ImT[t][group]*Rm)
        Vn[refr | fire] = V_reset
        if fire.any():
            spikes[fire, t] = True
            last[fire] = t
            Vth[fire] += step
        V = Vn
    return spikes, {'V': V, 'Vth': Vth, 'since': L - last}


def lif_window_with_adaptation(Im, Vth_base, V_reset, V_e, Rm, tau_m, dt, v_mem_start,
                               min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0,
                               state=None, return_state=False):
    """Single-unit LIF with threshold adaptation. Pass the `state` returned with
    return_state=True to continue the same unit across successive calls.
    Returns (spikes, V_end) or (spikes, V_end, state)."""
    if state is None:
        state = {'V': np.array([v_mem_start], float), 'Vth': np.array([Vth_base], float),
                 'since': np.array([10**9])}
    sp, st = lif_population(np.asarray(Im, float)[None, :], np.array([0]), np.array([Vth_base], float),
                            V_reset, V_e, Rm, tau_m, dt, state=state, adaptation=True,
                            min_isi_ms=min_isi_ms, dVth=dVth, tau_adapt_ms=tau_adapt_ms)
    out = (sp[0].astype(float), float(st['V'][0]))
    return out + (st,) if return_state else out

# =============================================================================
# Fibre dispersion
# =============================================================================

def simulate_fiber_emg_vectorized(unit_signal, n_fibers, delay_std_samp, rng, amp_mean=1.0, amp_std=0.1):
    L = len(unit_signal)
    if n_fibers <= 0:
        return np.zeros(L)
    delays = np.rint(rng.normal(0, delay_std_samp, n_fibers)).astype(int)
    amps = np.abs(rng.normal(amp_mean, amp_std, n_fibers))
    out = np.zeros(L)
    uniq, inv = np.unique(delays, return_inverse=True)
    sums = np.zeros_like(uniq, dtype=float)
    np.add.at(sums, inv, amps)
    for d, a in zip(uniq, sums):
        if d > 0:   out[d:] += a*unit_signal[:L-d]
        elif d < 0: out[:L+d] += a*unit_signal[-d:]
        else:       out += a*unit_signal
    return out

# =============================================================================
# Common drive and intent variability
# =============================================================================

def compose_common_drive(cortical, subcortical, cortical_weight, subcortical_weight,
                         target_rms=6.0, fraction_interpretation='amplitude'):
    """Mix cortical and subcortical common drive at a FIXED total RMS.

    cortical_weight : subcortical_weight sets the mixture (e.g. 6:0 = 100/0,
    4.875:1.625 = 75/25, 3.25:3.25 = 50/50); the mixed drive is then scaled to
    `target_rms`, so changing the mixture changes which frequencies are shared,
    not how much input is shared.

    fraction_interpretation='amplitude' (default): weights are amplitude
    weights (75/25 amplitude = ~90/10 in variance for independent sources).
    'power': weights are variance fractions.
    """
    c = zscore(cortical)
    s = zscore(subcortical)
    cw = abs(float(cortical_weight)); sw = abs(float(subcortical_weight))
    total = cw + sw
    if total <= 0:
        return np.zeros_like(c), dict(cortical_fraction=np.nan, rms=0.0, coefficients=(0.0, 0.0))
    p = cw / total
    if fraction_interpretation == 'amplitude':
        a, b = p, 1.0-p
    elif fraction_interpretation == 'power':
        a, b = np.sqrt(p), np.sqrt(1.0-p)
    else:
        raise ValueError("fraction_interpretation must be 'amplitude' or 'power'")
    raw = a*c + b*s
    raw_sd = np.std(raw)
    out = np.zeros_like(raw) if raw_sd <= 1e-15 else raw * (float(target_rms)/raw_sd)
    scale = 0.0 if raw_sd <= 1e-15 else float(target_rms)/raw_sd
    return out, dict(cortical_fraction=p, rms=float(np.std(out)), coefficients=(a*scale, b*scale))


def _bounded_intent_variation(n, fs, rms_fraction, rng, cutoff_hz=0.25, clip_sd=3.0):
    """Smooth low-frequency (<= cutoff_hz) variation of the drive envelope with the
    requested fractional RMS (random 5-component Fourier series, detrended)."""
    n = int(n); rms_fraction = float(rms_fraction); fs = float(fs)
    if rms_fraction <= 0 or n <= 2:
        return np.zeros(n, float)
    duration = n/fs
    f_hi = max(float(cutoff_hz), 1.0/duration)
    f_lo = max(1.0/duration, min(0.05, f_hi/5.0))
    freqs = np.linspace(f_lo, f_hi, 5)
    amps = rng.standard_normal(freqs.size)
    phases = rng.uniform(0.0, 2*np.pi, freqs.size)
    t = np.arange(n)/fs
    y = np.zeros(n, float)
    for a0, f0, p0 in zip(amps, freqs, phases):
        y += a0*np.sin(2*np.pi*f0*t + p0)
    y = scipy_detrend(y, type='linear')
    y -= y.mean()
    sd = y.std()
    if sd <= 1e-15:
        return np.zeros(n, float)
    y *= rms_fraction/sd
    limit = abs(float(clip_sd))*rms_fraction
    if limit > 0 and np.max(np.abs(y)) > limit:
        y = np.clip(y, -limit, limit)
        y -= y.mean()
        sd = y.std()
        if sd > 0:
            scale = min(rms_fraction/sd, limit/(np.max(np.abs(y)) + 1e-15))
            y *= scale
    return y

# =============================================================================
# Simulator
# =============================================================================

def generate_modulated_EMG_physiological_upgraded(
    motorunits_max=72, n_fibers=346, FreqInputTotal=100, SNR=40, SNRE=40, SNRF=8,
    V_reset=-0.080, V_e=-0.075, dt=0.0002, T_end=18.0, const_current=10.0,
    common_mod_subcortical=1.625,
    freq_left=0.5, freq_right=1.0, phase_left=0.0, phase_right=np.pi/2,
    Intent_variability=0.01, window_size=None,
    cross_talk_R2L=0.20, cross_talk_L2R=0.01, common_mod_cortical=4.875,
    Vth_min=-0.055, Vth_max=-0.040, I_scale=1.5e-9, Rm=10e6, tau_m=10e-3,
    seed=1234, cortical_high_hz=60,
    # MUAP
    use_dog_muaps=True, muap_dur_s=0.018, muap_jitter_s=0.004,
    mu_gain_low=0.6, mu_gain_high=1.4,
    # LIF adaptation
    use_lif_adaptation=True, min_isi_ms=18.0, dVth=0.003, tau_adapt_ms=250.0,
    # Noise
    noise_type='pink', shaper_taps_mu=None, shaper_taps_emg=None,
    # Common drive, duration, intent
    burn_in_s=1.0, common_drive_target_rms=6.0, common_fraction_interpretation='amplitude',
    intent_drift='bounded', intent_cutoff_hz=0.25,
    return_spikes=False, progress=False
):
    """Simulate a bilateral (left/right) monopolar EMG pair.

    Model: each limb receives a unique low-pass drive plus a common drive shared by
    both limbs (cortical 13-`cortical_high_hz` Hz and subcortical 5-13 Hz components,
    mixed at fixed total RMS), multiplied by a task force envelope; a fraction of each
    limb's input is mixed into the other (cross-limb crosstalk). The input drives a
    pool of leaky integrate-and-fire motor units with threshold adaptation; spike
    trains are convolved with MUAP templates, dispersed over muscle fibres and
    summed, and noise is added. All parameters have defaults equal to the
    representative configuration of the article.

    Key parameters
    --------------
    T_end : float        returned duration (s); a separate `burn_in_s` is simulated first
    dt : float           simulation step (s); fs = 1/dt (default 5 kHz)
    motorunits_max : int motor units per limb
    const_current : float  amplitude (SD) of the unique, limb-specific drive
    common_mod_cortical, common_mod_subcortical : float
        relative amplitude weights of the cortical/subcortical common drive
        (6/0 = 100/0, 4.875/1.625 = 75/25, 3.25/3.25 = 50/50)
    common_drive_target_rms : float  total SD of the common drive (default 6)
    cortical_high_hz : float  upper cut-off of the cortical band (30 or 60)
    cross_talk_R2L, cross_talk_L2R : float in [0, 1]
        fraction of the other limb's input mixed into each limb
    freq_left, freq_right, phase_left, phase_right : force-envelope sinusoids
        (default 1:2 bimanual task, 0.5 and 1.0 Hz); use freq=0, phase=pi/2 for a
        constant envelope (isometric task)
    Intent_variability : float  fractional RMS of slow (<0.25 Hz) envelope variation
    SNR, SNRF, SNRE : float     SNR in dB of drive noise, MU-level noise, EMG noise
    Vth_min, Vth_max, V_reset, V_e, Rm, tau_m, I_scale : LIF parameters
    min_isi_ms, dVth, tau_adapt_ms : refractory period and threshold adaptation
    muap_dur_s, muap_jitter_s : mean MUAP duration and per-MU uniform jitter (s)
    mu_gain_low, mu_gain_high : MUAP amplitude ramp across the pool
    n_fibers : int              fibres per motor unit (temporal dispersion)
    noise_type : 'pink' | 'white' | 'shaped'
    seed : int                  identical seeds give identical outputs
    window_size : ignored (kept for compatibility with earlier scripts)

    Returns
    -------
    EMG_L, EMG_R, fs            monopolar EMG (use simulate_bipolar_emg_spatial for
                                surface bipolar signals)
    info (if return_spikes=True) dict with spike trains ('spikes_L', 'spikes_R',
                                bool n_MU x n_samples), MUAP durations, gains, and
                                common-drive metadata.
    """
    if dt <= 0 or T_end <= 0:
        raise ValueError('dt and T_end must be positive')
    if not (0 <= cross_talk_R2L <= 1 and 0 <= cross_talk_L2R <= 1):
        raise ValueError('cross-talk coefficients must lie in [0,1]')
    if cortical_high_hz <= 13:
        raise ValueError('cortical_high_hz must be > 13 Hz')
    fs = 1.0/dt
    if cortical_high_hz >= fs/2:
        raise ValueError('cortical_high_hz must be below Nyquist')

    n_out = int(round(float(T_end)/dt))
    n_burn = int(round(max(float(burn_in_s), 0.0)/dt))
    N = n_burn + n_out
    T = (np.arange(N) - n_burn)*dt          # t = 0 is the first returned sample
    nMU = int(motorunits_max)
    if nMU < 1:
        raise ValueError('motorunits_max must be >= 1')

    # Independent, reproducible random streams for each stage.
    ss = np.random.SeedSequence(seed)
    (ss_shared, ss_drive, ss_intent, ss_muap, ss_munoise,
     ss_fiber, ss_emgnoise) = ss.spawn(7)
    rng_shared = np.random.default_rng(ss_shared)
    rng_drive = np.random.default_rng(ss_drive)
    rng_intent = np.random.default_rng(ss_intent)

    def band_limited_unit(length, low, high, rng):
        x = rng.standard_normal(length)
        x = butter_filter_low(x, high, fs)
        x = butter_filter_high(x, low, fs)
        return zscore(x)

    # Common drive (shared by both limbs)
    cort_z = band_limited_unit(N, 13, cortical_high_hz, rng_shared)
    sub_z = band_limited_unit(N, 5, 13, rng_shared)
    common, common_meta = compose_common_drive(
        cort_z, sub_z, common_mod_cortical, common_mod_subcortical,
        target_rms=common_drive_target_rms,
        fraction_interpretation=common_fraction_interpretation)

    # Task envelopes: shared 0.2-2 Hz fluctuation + limb-specific slow intent variation
    envn = 0.15*band_limited_unit(N, 0.2, 2.0, rng_shared)
    force_L = force_modulation_sine(T, freq_left, phase_left)
    force_R = force_modulation_sine(T, freq_right, phase_right)
    if intent_drift == 'bounded':
        drift_L = _bounded_intent_variation(N, fs, Intent_variability, rng_intent,
                                            cutoff_hz=intent_cutoff_hz)
        drift_R = _bounded_intent_variation(N, fs, Intent_variability, rng_intent,
                                            cutoff_hz=intent_cutoff_hz)
    elif intent_drift == 'none':
        drift_L = drift_R = np.zeros(N)
    else:
        raise ValueError("intent_drift must be 'bounded' or 'none'")
    fL = force_L*(1.0 + envn + drift_L)
    fR = force_R*(1.0 + envn + drift_R)

    # Unique drive + common drive + drive noise
    uL = zscore(butter_filter_low(rng_drive.standard_normal(N), FreqInputTotal, fs))
    uR = zscore(butter_filter_low(rng_drive.standard_normal(N), FreqInputTotal, fs))
    sigL = const_current*uL + common
    sigR = const_current*uR + common
    nL0 = zscore(rng_drive.standard_normal(N))
    nR0 = zscore(rng_drive.standard_normal(N))
    nL = nL0*_snr_noise_sd(np.std(sigL), SNR)
    nR = nR0*_snr_noise_sd(np.std(sigR), SNR)
    ImL0 = I_scale*(sigL + nL)*fL
    ImR0 = I_scale*(sigR + nR)*fR

    # Cross-limb crosstalk (convex mixing)
    ImL = (1.0-cross_talk_R2L)*ImL0 + cross_talk_R2L*ImR0
    ImR = (1.0-cross_talk_L2R)*ImR0 + cross_talk_L2R*ImL0

    # Motor-neuron pools, both limbs, continuous integration over burn-in + data
    Th = np.linspace(Vth_min, Vth_max, nMU)
    spikes, _ = lif_population(
        np.vstack([ImL, ImR]),
        np.r_[np.zeros(nMU, int), np.ones(nMU, int)],
        np.r_[Th, Th], V_reset, V_e, Rm, tau_m, dt,
        adaptation=use_lif_adaptation, min_isi_ms=min_isi_ms,
        dVth=dVth, tau_adapt_ms=tau_adapt_ms)
    spL, spR = spikes[:nMU], spikes[nMU:]

    # MUAP synthesis, MU-level noise, fibre dispersion
    rng_muap = np.random.default_rng(ss_muap)
    durs = muap_dur_s + rng_muap.uniform(-muap_jitter_s, muap_jitter_s, nMU)
    gains = np.linspace(mu_gain_low, mu_gain_high, nMU)
    rng_mun = np.random.default_rng(ss_munoise)
    fiber_ss = ss_fiber.spawn(2*nMU)
    EMG_L = np.zeros(N); EMG_R = np.zeros(N)
    it = tqdm(range(nMU), desc='MUAPs', leave=False) if progress else range(nMU)
    for i in it:
        mu_factor = i/(nMU-1 + 1e-12)
        shape = (generate_muap_shape_dog(durs[i], fs, mu_factor=mu_factor)
                 if use_dog_muaps else generate_muap_shape_original(durs[i], dt))
        for side, sp, EMG in ((0, spL[i], EMG_L), (1, spR[i], EMG_R)):
            conv = np.convolve(sp.astype(float), shape, mode='same')*gains[i]
            sd = np.std(conv)
            if sd > 0:
                nz = _noise(N, fs, noise_type, rng_mun, shaper_taps_mu)
                conv += nz*_snr_noise_sd(sd, SNRF)
            rng_f = np.random.default_rng(fiber_ss[2*i + side])
            EMG += simulate_fiber_emg_vectorized(conv, n_fibers, delay_std_samp=1, rng=rng_f)

    # Discard burn-in, add EMG-level noise
    EMG_L = EMG_L[n_burn:n_burn+n_out]
    EMG_R = EMG_R[n_burn:n_burn+n_out]
    rng_e = np.random.default_rng(ss_emgnoise)

    def add_emg_noise(x):
        sd = np.std(x)
        if sd <= 0:
            return x
        nz = _noise(x.size, fs, noise_type, rng_e, shaper_taps_emg)
        return x + nz*_snr_noise_sd(sd, SNRE)

    EMG_L = add_emg_noise(EMG_L)
    EMG_R = add_emg_noise(EMG_R)

    if return_spikes:
        info = dict(
            spikes_L=spL[:, n_burn:n_burn+n_out],
            spikes_R=spR[:, n_burn:n_burn+n_out],
            muap_dur_s=durs, gains=gains,
            burn_in_s=n_burn*dt, duration_s=n_out*dt,
            common_drive_rms=common_meta['rms'],
            common_cortical_fraction=common_meta['cortical_fraction'],
            common_effective_coefficients=common_meta['coefficients'],
            drive_noise_snr_db_L=measured_snr_db(sigL, nL),
            drive_noise_snr_db_R=measured_snr_db(sigR, nR),
        )
        return EMG_L, EMG_R, fs, info
    return EMG_L, EMG_R, fs


def simulate_bilateral_emg(seed=1234, bipolar=True, fs_out=None, **params):
    """Convenience wrapper: simulate a left/right EMG pair with the article's
    representative configuration (1:2 bimanual task, 75/25 cortical/subcortical
    common drive, cortical band 13-60 Hz, 20 % right-to-left crosstalk, 72 motor
    units per limb, 18 s). Any argument of
    generate_modulated_EMG_physiological_upgraded can be overridden via **params.

    bipolar : return surface bipolar signals (default) instead of monopolar
    fs_out  : resample the output to this rate (e.g. 1000.0); None keeps 1/dt

    Returns left, right, fs  (plus info if return_spikes=True).
    """
    out = generate_modulated_EMG_physiological_upgraded(seed=seed, **params)
    L, R, fs = out[:3]
    if bipolar:
        L = simulate_bipolar_emg_spatial(L, fs, tau_ms=1.2, lp_hz=180.0)
        R = simulate_bipolar_emg_spatial(R, fs, tau_ms=1.2, lp_hz=180.0)
    if fs_out is not None:
        L = resample_to_fs(L, fs, fs_out); R = resample_to_fs(R, fs, fs_out); fs = float(fs_out)
    return (L, R, fs) + tuple(out[3:])

# =============================================================================
# Preprocessing for coherence
# =============================================================================

def apply_analysis_taper(x, taper=None, taper_kwargs=None, zscore_output=True):
    """Optional taper applied to a copy of an already preprocessed signal.
    taper: None (default), 'matviyenko' (taper_kwargs={'K': 2..13}), 'tukey'
    (taper_kwargs={'alpha': ...}), a callable returning a length-N window, or an array."""
    y = np.asarray(x, float).copy()
    if taper is not None:
        kw = {} if taper_kwargs is None else dict(taper_kwargs)
        if isinstance(taper, str):
            taper_name = taper.lower()
            if taper_name == 'matviyenko':
                K = int(kw.pop('K', 8))
                if kw:
                    raise TypeError(f"Unexpected Matviyenko taper kwargs: {sorted(kw)}")
                w = matviyenko_window(y.size, K=K)
            elif taper_name == 'tukey':
                w = signal_windows.tukey(y.size, alpha=float(kw.pop('alpha', 0.1)))
                if kw:
                    raise TypeError(f"Unexpected Tukey taper kwargs: {sorted(kw)}")
            else:
                raise ValueError("Unknown taper. Use None, 'matviyenko', 'tukey', callable, or an array.")
        elif callable(taper):
            w = np.asarray(taper(y.size, **kw), float)
        else:
            w = np.asarray(taper, float)
        if w.shape != y.shape:
            raise ValueError('taper must have the same length as the signal')
        y *= w
    return zscore(y) if zscore_output else y


def preprocess_emg_for_coherence(x, fs, low=5.0, high=499.0, detrend=True,
                                 notch_hz=None, notch_harmonics=3,
                                 taper=None, taper_kwargs=None, zscore_output=True):
    """Prepare NON-rectified EMG for wavelet coherence: optional power-line notch
    (notch_hz = 50 or 60), band-pass `low`-`high` Hz (4th-order zero-phase
    Butterworth), linear detrend, optional taper (default none) and z-score."""
    y = np.asarray(x, float)
    if notch_hz is not None:
        y = notch_filter(y, fs, f0=notch_hz, harmonics=notch_harmonics)
    y = bandpass_emg(y, fs, low=low, high=min(high, fs/2.0-1e-6))
    if detrend:
        y = scipy_detrend(y, type='linear')
    return apply_analysis_taper(y, taper=taper, taper_kwargs=taper_kwargs,
                                zscore_output=zscore_output)

# =============================================================================
# Wavelet coherence (Morlet; Torrence & Compo 1998; Grinsted et al. 2004)
# =============================================================================

def _require_pycwt():
    if wavelet is None:
        raise ImportError("Wavelet analysis requires 'pycwt' (pip install -r requirements.txt).")


def smoothwavelet(wave, dt, period, dj, scale):
    """Smoothing operator for wavelet coherence: Gaussian along time (width
    proportional to scale) and a 0.6-octave boxcar along scale (Morlet)."""
    wave = np.asarray(wave)
    if wave.ndim != 2:
        raise ValueError("wave must be a 2-D (scale, time) array")
    n = wave.shape[1]
    npad = 2 ** int(np.ceil(np.log2(n)))
    k = np.arange(1, npad//2 + 1) * (2*np.pi/npad)
    k = np.concatenate(([0.], k, -k[-2::-1]))
    k2 = k**2
    snorm = np.asarray(scale, float)/float(dt)
    F = np.exp(-0.5*(snorm[:, None]**2)*k2[None, :])
    twave = np.fft.ifft(F*np.fft.fft(wave, npad, axis=1), axis=1)[:, :n]
    if np.isrealobj(wave):
        twave = twave.real
    dj0 = 0.6
    dj0steps = dj0/(dj*2)
    kernel = np.concatenate(([dj0steps % 1], np.ones(int(2*round(dj0steps)-1)), [dj0steps % 1]))
    kernel /= kernel.sum()
    return convolve2d(twave, kernel[:, np.newaxis], mode="same")


def _wct_from_cwt(X, Y, scales, dt, dj):
    sinv = 1.0/scales
    sX = smoothwavelet(sinv[:, None]*(np.abs(X)**2), dt, None, dj, scales)
    sY = smoothwavelet(sinv[:, None]*(np.abs(Y)**2), dt, None, dj, scales)
    sWxy = smoothwavelet(sinv[:, None]*(X*np.conj(Y)), dt, None, dj, scales)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rsq = np.abs(sWxy)**2/(sX*sY)
    Rsq = np.clip(np.real(Rsq), 0.0, 1.0)
    Rsq[~np.isfinite(Rsq)] = np.nan
    return Rsq


def _default_J(fs, dj, s0, w0, fmin):
    """Number of scales so that the lowest analysed frequency is <= fmin."""
    fourier_factor = 4*np.pi/(w0 + np.sqrt(2 + w0**2))
    f_top = 1.0/(fourier_factor*s0)
    return int(np.ceil(np.log2(f_top/float(fmin))/dj))


def compute_cwt_for_coherence(sig, fs, fmax=128.0, dj=1/8, s0=None, J=None, w0=6, fmin=4.0):
    """Morlet CWT of one signal, packaged for coherence_from_cached_cwts
    (lets many pairings reuse the same transform)."""
    sig = np.asarray(sig, float)
    if sig.ndim != 1:
        raise ValueError('sig must be a 1-D array')
    dt = 1.0/float(fs)
    if s0 is None:
        s0 = dt
    J = _default_J(fs, dj, s0, w0, fmin) if J is None else int(J)
    _require_pycwt()
    mother = wavelet.Morlet(w0)
    W, scales, freqs_full, coi, *_ = wavelet.cwt(sig, dt, dj, s0, J, mother)
    fmask = freqs_full <= fmax
    return dict(
        W=W, scales=np.asarray(scales, float), freqs_full=np.asarray(freqs_full, float),
        freqs=np.asarray(freqs_full[fmask], float), period=np.asarray(1.0/freqs_full[fmask], float),
        coi=np.asarray(coi, float), fmask=np.asarray(fmask, bool), dt=float(dt), dj=float(dj),
        s0=float(s0), J=int(J), w0=float(w0), mother=mother, fs=float(fs), fmax=float(fmax),
    )


def coherence_from_cached_cwts(left, right):
    """Squared wavelet coherence from two outputs of compute_cwt_for_coherence.
    Returns freqs, period, coi, Rsq (n_freqs x n_times)."""
    for key in ('dt', 'dj', 's0', 'J', 'w0', 'fs'):
        if not np.isclose(left[key], right[key]):
            raise ValueError(f'CWT grids differ ({key})')
    if not np.allclose(left['scales'], right['scales']):
        raise ValueError('CWT scales differ')
    if not np.array_equal(left['fmask'], right['fmask']):
        raise ValueError('CWT frequency masks differ')
    Rsq_full = _wct_from_cwt(left['W'], right['W'], left['scales'], left['dt'], left['dj'])
    m = left['fmask']
    coi = np.minimum(left['coi'], right['coi'])
    return left['freqs'], left['period'], coi, Rsq_full[m, :]


def compute_wavelet_coherence(sig1, sig2, fs, fmax=128.0, dj=1/8, s0=None, J=None, w0=6, fmin=4.0):
    """Squared Morlet wavelet coherence of two equal-length signals.

    Inputs should be preprocessed (preprocess_emg_for_coherence). Frequencies from
    about `fmin` to `fmax` Hz are analysed (dj = scale resolution in octaves).

    Returns freqs, period, coi, Rsq, dt, dj, s0, J, mother
        Rsq has shape (len(freqs), len(sig1)); coi is the cone of influence (in
        period units) for each time sample. Summarise with
        band_medians(Rsq, freqs, period=period, coi=coi).
    """
    sig1 = np.asarray(sig1, float); sig2 = np.asarray(sig2, float)
    if sig1.ndim != 1 or sig1.shape != sig2.shape:
        raise ValueError('sig1 and sig2 must be equal-length 1-D arrays')
    L = compute_cwt_for_coherence(sig1, fs, fmax=fmax, dj=dj, s0=s0, J=J, w0=w0, fmin=fmin)
    R = compute_cwt_for_coherence(sig2, fs, fmax=fmax, dj=dj, s0=s0, J=J, w0=w0, fmin=fmin)
    freqs, period, coi, Rsq = coherence_from_cached_cwts(L, R)
    return freqs, period, coi, Rsq, L['dt'], L['dj'], L['s0'], L['J'], L['mother']


def band_medians(Rsq, freqs, bands=DEFAULT_BANDS, period=None, coi=None):
    """Median squared coherence in each frequency band.

    Pass `period` and `coi` (from compute_wavelet_coherence) to use only
    time-frequency points inside the cone of influence (recommended).
    Returns a dict {'5-13': value, ...}.
    """
    Rsq = np.asarray(Rsq, float)
    freqs = np.asarray(freqs, float)
    if Rsq.ndim != 2 or Rsq.shape[0] != freqs.size:
        raise ValueError("Rsq must have shape (len(freqs), n_times)")
    use_coi = period is not None and coi is not None
    if use_coi:
        period = np.asarray(period, float)
        coi = np.asarray(coi, float)
        if period.size != freqs.size or coi.size != Rsq.shape[1]:
            raise ValueError("period/coi dimensions do not match Rsq")
    out = {}
    for (f0, f1) in bands:
        idx = np.where((freqs >= f0) & (freqs < f1))[0]
        if idx.size == 0:
            out[f"{f0}-{f1}"] = np.nan
            continue
        values = Rsq[idx, :]
        values = values[period[idx, None] <= coi[None, :]] if use_coi else values.ravel()
        values = values[np.isfinite(values)]
        out[f"{f0}-{f1}"] = float(np.median(values)) if values.size else np.nan
    return out

def plot_coherence_map(Rsq, freqs, period, coi, fs, path=None, title="Wavelet coherence"):
    """Time-frequency coherence map with the cone of influence (outside shaded).
    Saves to `path` if given; returns the matplotlib figure."""
    import matplotlib.pyplot as plt
    t = np.arange(Rsq.shape[1])/fs
    fig, ax = plt.subplots(figsize=(12, 5))
    cf = ax.contourf(t, np.log2(period), Rsq, levels=np.linspace(0, 1, 41), cmap="jet", vmin=0, vmax=1)
    ax.plot(t, np.log2(coi), "k", lw=2, label="cone of influence")
    ax.fill_between(t, np.log2(coi), np.log2(period.max()), color="w", alpha=0.4)
    ticks = np.array([100, 60, 30, 13, 5])
    ax.set_yticks(np.log2(1.0/ticks)); ax.set_yticklabels(ticks)
    ax.set_ylim(np.log2(1/4.0), np.log2(1/min(100.0, freqs.max())))
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title)
    fig.colorbar(cf, ax=ax, pad=0.02, label="Squared coherence")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200); plt.close(fig)
    return fig


# =============================================================================
# Matched cross-trial null (significance of coherence)
# =============================================================================

def _n_derangements(n):
    d = [1, 0]
    for k in range(2, n + 1):
        d.append((k - 1)*(d[-1] + d[-2]))
    return d[n]


def random_derangement(n, rng=None):
    """Random permutation of range(n) with no fixed points (n >= 2)."""
    n = int(n)
    if n < 2:
        raise ValueError('a derangement requires at least two trials')
    rng = np.random.default_rng() if rng is None else rng
    base = np.arange(n)
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != base):
            return p
    shift = int(rng.integers(1, n))
    return np.roll(base, shift)


def cross_trial_null(left_trials, right_trials, fs, bands=DEFAULT_BANDS, n_perm=999,
                     alpha=0.05, rng=None, fmax=128.0, dj=1/8, w0=6, fmin=4.0,
                     progress=True):
    """Test whether left-right coherence exceeds a matched cross-trial null.

    The null pairs the left signal of trial i with the right signal of a
    different trial j of the same condition. This keeps each signal's spectrum,
    amplitude modulation and task timing but removes any within-trial coupling.
    The test statistic is the median (across trials) of the band-median
    coherence; the null distribution is built from random derangements of the
    trial labels.

    Parameters
    ----------
    left_trials, right_trials : sequences of equal-length 1-D arrays (one per trial),
        already preprocessed (preprocess_emg_for_coherence). At least 2 trials are
        required; >= 6 are recommended (with n trials the smallest possible p is
        1/(D_n + 1), D_n = number of derangements: 2 -> 1/2, 4 -> 1/10, 6 -> 1/266).
        For n <= 8 with D_n <= n_perm all derangements are used (exact test).

    Returns
    -------
    dict with, per band (arrays ordered as `bands`):
        'bands', 'observed' (median over trials of within-trial coherence),
        'per_trial' (n_trials x n_bands), 'null_median', 'null_95' (1-alpha
        quantile), 'excess' (observed - null_median), 'p' (one-sided).
    """
    left_trials = [np.asarray(x, float) for x in left_trials]
    right_trials = [np.asarray(y, float) for y in right_trials]
    n = len(left_trials)
    if n < 2 or len(right_trials) != n:
        raise ValueError('need the same number (>= 2) of left and right trials')
    rng = np.random.default_rng() if rng is None else rng
    kw = dict(fmax=fmax, dj=dj, w0=w0, fmin=fmin)
    Lc = [compute_cwt_for_coherence(x, fs, **kw) for x in left_trials]
    Rc = [compute_cwt_for_coherence(y, fs, **kw) for y in right_trials]
    nb = len(bands)
    pair = np.full((n, n, nb), np.nan)
    it = tqdm(range(n), desc='trial pairs', leave=False) if progress else range(n)
    for i in it:
        for j in range(n):
            f, p, coi, R = coherence_from_cached_cwts(Lc[i], Rc[j])
            pair[i, j] = list(band_medians(R, f, bands, period=p, coi=coi).values())
    per_trial = pair[np.arange(n), np.arange(n)]
    observed = np.nanmedian(per_trial, axis=0)
    # Few trials: use every derangement (exact test). Otherwise sample n_perm of them.
    n_der = _n_derangements(n)
    if n <= 8 and n_der <= int(n_perm):
        perms = [np.array(q) for q in itertools.permutations(range(n))
                 if all(q[i] != i for i in range(n))]
    else:
        perms = [random_derangement(n, rng) for _ in range(int(n_perm))]
    null = np.array([np.nanmedian(pair[np.arange(n), q], axis=0) for q in perms])
    n_perm = len(perms)
    return dict(
        bands=[f"{a}-{b}" for a, b in bands],
        observed=observed, per_trial=per_trial,
        null_median=np.nanmedian(null, axis=0),
        null_95=np.nanpercentile(null, 100*(1-alpha), axis=0),
        excess=observed - np.nanmedian(null, axis=0),
        p=(1 + np.sum(null >= observed[None, :], axis=0))/(n_perm + 1),
        n_trials=n, n_perm=int(n_perm),
    )
