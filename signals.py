"""Signal + jammer generation and spectrogram conversion.

Everything is complex baseband at FS = 1 MHz, N = 4096 samples.
The spectrogram uses 64-sample segments with no overlap, so it is
exactly 64 (freq) x 64 (time) with no resizing.
"""
import numpy as np
from scipy import signal as sps

FS = 1_000_000          # sample rate, Hz
N = 4096                # samples per example
NFFT = 64               # spectrogram segment length -> 64 freq bins, 64 time bins
SPS = 4                 # samples per QPSK symbol (~250 kHz occupied bandwidth)

TRAIN_CLASSES = ["clean", "tone", "wideband", "sweep"]
UNSEEN_CLASSES = ["pulsed"]
ALL_KINDS = TRAIN_CLASSES + UNSEEN_CLASSES

SNR_RANGE = (5.0, 20.0)
JSR_RANGE = (-10.0, 20.0)
FMAX = 0.45 * FS        # keep jammers inside +/-450 kHz

# Pulsed jammer ranges. Fixed before any results were seen; never tuned on outcomes.
PULSE_PERIOD = (256, 1024)   # samples (4 to 16 spectrogram time bins)
PULSE_DUTY = (0.10, 0.60)

_SHAPING = sps.firwin(33, 1.0 / SPS)  # windowed-sinc pulse shaping filter
_SHAPING /= np.sqrt(np.sum(_SHAPING ** 2))


def _unit_power(x):
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


def qpsk(rng):
    n_sym = N // SPS + 16
    sym = (rng.choice([-1, 1], n_sym) + 1j * rng.choice([-1, 1], n_sym)) / np.sqrt(2)
    up = np.zeros(n_sym * SPS, complex)
    up[::SPS] = sym
    x = np.convolve(up, _SHAPING, mode="same")[:N]
    return _unit_power(x)


def awgn(rng, power):
    return np.sqrt(power / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))


def _t():
    return np.arange(N) / FS


def jam_tone(rng):
    f = rng.uniform(-FMAX, FMAX)
    return np.exp(1j * (2 * np.pi * f * _t() + rng.uniform(0, 2 * np.pi)))


def jam_wideband(rng):
    bw = rng.uniform(30e3, 300e3)
    fc = rng.uniform(-FMAX + bw / 2, FMAX - bw / 2)
    spec = np.fft.fft(rng.standard_normal(N) + 1j * rng.standard_normal(N))
    freqs = np.fft.fftfreq(N, 1 / FS)
    spec[np.abs(freqs - fc) > bw / 2] = 0
    return _unit_power(np.fft.ifft(spec))


def jam_sweep(rng):
    f0, f1 = rng.uniform(-FMAX, FMAX, 2)
    while abs(f1 - f0) < 100e3:
        f0, f1 = rng.uniform(-FMAX, FMAX, 2)
    n_sweeps = rng.integers(1, 5)               # sawtooth with 1 to 4 sweeps per example
    period = N / FS / n_sweeps
    t = (_t() + rng.uniform(0, period)) % period
    re = sps.chirp(t, f0=f0, t1=period, f1=f1, method="linear", phi=0)
    im = sps.chirp(t, f0=f0, t1=period, f1=f1, method="linear", phi=-90)
    return re + 1j * im


def jam_pulsed(rng):
    period = rng.integers(*PULSE_PERIOD)
    duty = rng.uniform(*PULSE_DUTY)
    offset = rng.integers(0, period)
    gate = (((np.arange(N) + offset) % period) < duty * period).astype(float)
    return jam_tone(rng) * gate                  # JSR = power while the pulse is on


JAMMERS = {"tone": jam_tone, "wideband": jam_wideband, "sweep": jam_sweep, "pulsed": jam_pulsed}


def to_spectrogram(iq):
    """Two-sided log-magnitude spectrogram, 64x64 (freq x time), per-example z-scored.
    Row 0 is the most negative frequency."""
    _, _, S = sps.spectrogram(iq, fs=FS, window="hann", nperseg=NFFT, noverlap=0,
                              return_onesided=False, mode="psd")
    S = np.fft.fftshift(S, axes=0)
    S = 10 * np.log10(S + 1e-12)
    return ((S - S.mean()) / (S.std() + 1e-6)).astype(np.float32)


def generate(kind, jsr_db=None, seed=None, snr_db=None, return_meta=False):
    """Returns (iq, spectrogram). JSR/SNR are random unless given."""
    rng = np.random.default_rng(seed)
    snr_db = rng.uniform(*SNR_RANGE) if snr_db is None else snr_db
    jsr_db = rng.uniform(*JSR_RANGE) if jsr_db is None else jsr_db
    x = qpsk(rng) + awgn(rng, 10 ** (-snr_db / 10))
    if kind != "clean":
        x = x + np.sqrt(10 ** (jsr_db / 10)) * JAMMERS[kind](rng)
    if return_meta:
        return x, to_spectrogram(x), {"snr_db": float(snr_db), "jsr_db": float(jsr_db)}
    return x, to_spectrogram(x)


def spectrogram_axes():
    """Frequency (kHz) and time (ms) extents for plotting: (t0, t1, f0, f1)."""
    return 0.0, N / FS * 1e3, -FS / 2e3, FS / 2e3
