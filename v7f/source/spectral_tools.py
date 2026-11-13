"""
spectral_tools.py - the spectral pipeline used for the athletes' EMG, applied
identically to simulated EMG.

EMG at 1 kHz -> 60/120/180 Hz notch (Q = 30) + 10-450 Hz band-pass (4th-order
Butterworth, zero phase) -> Welch PSD (nperseg 512) -> 10-300 Hz.
Notch gaps (+-6 Hz) are interpolated before computing summary measures.

  mdf(P)                   median frequency
  shape_indices(P)         scale-free shape: (F75-F25)/F50, MNF/MDF, % power below
                           0.5 x MDF, % power between 0.5 and 1 x MDF
  rescale(M, mdfs)         spectra on the u = f/MDF axis, unit area (Jacobian)
  collapse(Bn, En, ...)    % of the begin-vs-end difference removed by rescaling
  cluster_test(D)          cluster-mass sign-flip permutation test (FWE)
"""
import numpy as np
from scipy import stats
from scipy.signal import welch, iirnotch, filtfilt, butter

FS = 1000.0
NPERSEG, FMIN, FMAX = 512, 10.0, 300.0
XG = np.linspace(0.3, 2.5, 120)                 # u = f / MDF grid
FG = np.fft.rfftfreq(NPERSEG, 1/FS)
FG = FG[(FG >= FMIN) & (FG <= FMAX)]
GAP = np.zeros_like(FG, bool)
for _f0 in (60, 120, 180):
    GAP |= (FG > _f0 - 6) & (FG < _f0 + 6)


def prep(x, fs=FS, lo=10.0, hi=450.0):
    x = np.asarray(x, float)
    for f0 in (60, 120, 180):
        b, a = iirnotch(f0, 30, fs)
        x = filtfilt(b, a, x)
    b, a = butter(4, [lo/(fs/2), min(hi, fs/2*0.99)/(fs/2)], btype="band")
    return filtfilt(b, a, x)


def psd(x, fs=FS):
    """PSD on the common 10-300 Hz grid FG (raw, notch gaps not interpolated)."""
    f, P = welch(prep(x, fs), fs=fs, nperseg=min(NPERSEG, len(x)))
    m = (f >= FMIN) & (f <= FMAX)
    return np.interp(FG, f[m], P[m])


def denotch(P):
    """Interpolate across the +-6 Hz notch gaps (1-D spectrum or participants x FG)."""
    A = np.atleast_2d(np.array(P, float))
    for r in A:
        r[GAP] = np.interp(FG[GAP], FG[~GAP], r[~GAP])
    return A if np.ndim(P) == 2 else A[0]


def qfreq(P, q):
    c = np.cumsum(P)
    return float(np.interp(q*c[-1], c, FG))


def mdf(P):
    return qfreq(P, 0.5)


SHAPE_NAMES = ["rel_bandwidth", "MNF_over_MDF", "share_below_half_MDF", "share_half_to_1_MDF"]


def shape_indices(P):
    f25, f50, f75 = (qfreq(P, q) for q in (0.25, 0.5, 0.75))
    mnf = np.sum(FG*P)/np.sum(P)
    lo = 100*np.sum(P[FG < 0.5*f50])/np.sum(P)
    mid = 100*np.sum(P[(FG >= 0.5*f50) & (FG < f50)])/np.sum(P)
    return np.array([(f75 - f25)/f50, mnf/f50, lo, mid])


def rescale(M, mdfs):
    out = []
    for r, m in zip(np.atleast_2d(M), mdfs):
        y = np.interp(XG, FG/m, r, left=0, right=0)
        out.append(y/(np.trapezoid(y, XG) + 1e-30)*100)
    return np.array(out)


def collapse(B, E):
    """B, E: de-notched spectra (participants x FG). Returns (raw gap, residual gap, % removed)."""
    nb, ne = B/B.sum(1, keepdims=True), E/E.sum(1, keepdims=True)
    mb, me = np.median(nb, 0), np.median(ne, 0)
    raw = np.trapezoid(np.abs(mb - me), FG)/np.trapezoid(0.5*(mb + me), FG)
    rb = rescale(B, [mdf(x) for x in B]); re_ = rescale(E, [mdf(x) for x in E])
    Rb, Re = np.median(rb, 0), np.median(re_, 0)
    res = np.trapezoid(np.abs(Rb - Re), XG)/np.trapezoid(0.5*(Rb + Re), XG)
    return raw, res, 100*(1 - res/raw) if raw > 0 else np.nan


def cluster_test(D, nperm=2000, alpha=0.05, rng=None):
    """Paired cluster-mass test on D (participants x bins, e.g. end - begin on the u axis).
    Returns [(u_start, u_end, mass, p_FWE)]."""
    rng = np.random.default_rng(0) if rng is None else rng
    n = D.shape[0]
    tcrit = stats.t.ppf(1 - alpha/2, n - 1)

    def tvals(X):
        return X.mean(0)/(X.std(0, ddof=1)/np.sqrt(n) + 1e-30)

    def clusters(t):
        out, j, s = [], 0, np.sign(t)*(np.abs(t) > tcrit)
        while j < len(t):
            if s[j]:
                k = j
                while k + 1 < len(t) and s[k + 1] == s[j]:
                    k += 1
                out.append((j, k, float(t[j:k + 1].sum()))); j = k + 1
            else:
                j += 1
        return out
    obs = clusters(tvals(D))
    null = np.array([max([abs(c[2]) for c in clusters(tvals(D*rng.choice([-1, 1], (n, 1))))], default=0.0)
                     for _ in range(nperm)])
    return [(XG[a], XG[b], m, float((np.sum(null >= abs(m)) + 1)/(nperm + 1))) for a, b, m in obs]
