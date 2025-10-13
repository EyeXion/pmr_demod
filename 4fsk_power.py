import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from threading import Thread
from collections import deque
from datetime import datetime
import time
from math import ceil, atan2, log2
from fractions import Fraction

# ------------------------------------------------------------------------------
# 1) SDR & Burst-Detection Parameters
# ------------------------------------------------------------------------------
SAMPLE_RATE = 4.8e6
CENTER_FREQ = 433.000e6
DECIMATION = int(SAMPLE_RATE / 48e3)

MAX_BURST_S = 0.1  # maximum burst length [s]
PREAMBLE_S = 0.010  # keep 10 ms of preamble
THRESH_FACTOR = 5.0  # envelope threshold
ENV_FILT_CUTOFF = 1e4  # lowpass cutoff for envelope smoothing

# ring buffer length to hold PREAMBLE_S + MAX_BURST_S
RINGBUF_SIZE = int((PREAMBLE_S + MAX_BURST_S) * SAMPLE_RATE)

# ------------------------------------------------------------------------------
# 2) RRC & Symbol-Mapping from your existing code
# ------------------------------------------------------------------------------
RRCOS_FILTER = [...]  # (same coefficients)
SYMBOLS = np.array([0, -3, -1, 1, 3], dtype=np.int32)
_SYMBOL_TO_BITS = {3: "01", 1: "00", -1: "10", -3: "11"}


def find_best_phase_offset_and_std(instant_frequencies, sps):
    L = (len(instant_frequencies) // sps) * sps
    arr = instant_frequencies[:L].reshape(-1, sps).T
    stds = np.sqrt(arr.var(axis=1))
    best = int(np.argmax(stds))
    return best, float(stds[best])


def create_frequency_bins(instant_frequencies):
    p05, p95 = np.quantile(instant_frequencies, [0.05, 0.95])
    mid = (p05 + p95) * 0.5
    bins = np.array(
        [
            instant_frequencies.min() - 1,
            (instant_frequencies.min() + mid) * 0.5,
            mid,
            (instant_frequencies.max() + mid) * 0.5,
            instant_frequencies.max() + 1,
        ]
    )
    return bins


def frequencies_to_symbols(instant_frequencies, bins):
    idx = np.digitize(instant_frequencies, bins, right=True)
    idx = np.clip(idx, 1, len(bins) - 1)
    syms = SYMBOLS[idx]
    left, right = bins[:-1], bins[1:]
    mids = (left + right) * 0.5
    hws = (right - left) * 0.5
    mid = mids[idx - 1]
    hw = hws[idx - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cert = 1.0 - np.abs(instant_frequencies - mid) / hw
    cert = np.where(hw == 0, 0.0, cert)
    cert = np.clip(cert, 0, 1)
    out = np.empty(
        instant_frequencies.shape, dtype=[("symbol", "i4"), ("certainty", "f8")]
    )
    out["symbol"] = syms
    out["certainty"] = cert
    return out


def symbols_to_hex(symbols):
    bits = "".join(_SYMBOL_TO_BITS[int(s)] for s in symbols)
    pad = (-len(bits)) % 8
    bits += "0" * pad
    b_arr = [int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)]
    return "".join(f"{b:02x}" for b in b_arr)


# ------------------------------------------------------------------------------
# 3) Demod Pipeline (reuse your update() body minus plotting)
# ------------------------------------------------------------------------------
def demodulate_iq(iq_samples, fs):
    # Frequency translate to audio center
    t = np.arange(len(iq_samples)) / fs
    shift = CENTER_FREQ - 433.5815e6
    iq = iq_samples * np.exp(1j * 2 * np.pi * shift * t)

    # Lowpass + decimate to 48 kHz
    sos = signal.butter(5, 9.6e3, fs=fs, output="sos", btype="low")
    flt = signal.sosfilt(sos, iq)
    dec = signal.decimate(flt, DECIMATION, ftype="fir")
    fs2 = fs / DECIMATION

    # Instantaneous phase & freq
    iphase = np.unwrap(np.angle(dec))
    ifreq = np.diff(iphase)
    ifreq = signal.lfilter(RRCOS_FILTER, 1, ifreq)
    sps = int(fs2 // 4800)

    # Symbol timing
    offset, _ = find_best_phase_offset_and_std(ifreq, sps)
    samp_freqs = ifreq[offset::sps]

    # Symbol decision
    bins = create_frequency_bins(samp_freqs)
    out = frequencies_to_symbols(samp_freqs, bins)
    hexstr = symbols_to_hex(out["symbol"])
    return samp_freqs, bins, out, hexstr


# ------------------------------------------------------------------------------
# 4) Plot Setup & Slider (same as your original)
# ------------------------------------------------------------------------------
window_duration = 0.0275
window_offset = 0.0

fig, axes = plt.subplots(3, 1)
plt.subplots_adjust(left=0.15, right=0.85)

ax_wd = fig.add_axes([0.05, 0.25, 0.01, 0.65])
sld_wd = Slider(
    ax_wd,
    "Win [s]",
    valmin=1 / 4800,
    valmax=MAX_BURST_S,
    valstep=0.001,
    valinit=window_duration,
    orientation="vertical",
)
ax_wo = fig.add_axes([0.95, 0.25, 0.01, 0.65])
sld_wo = Slider(
    ax_wo,
    "Off [s]",
    valmin=0,
    valmax=MAX_BURST_S,
    valstep=1 / 9600,
    valinit=window_offset,
    orientation="vertical",
)


def update_plot(_=None):
    global window_duration, window_offset, burst_iq, burst_fs2
    if len(burst_iq) == 0:
        return

    window_duration = sld_wd.val
    window_offset = min(sld_wo.val, len(burst_iq) / SAMPLE_RATE - window_duration)
    sld_wo.valmax = len(burst_iq) / SAMPLE_RATE - window_duration
    sld_wo.ax.set_xlim(0, sld_wo.valmax)

    start = int(window_offset * SAMPLE_RATE)
    Nw = int(window_duration * SAMPLE_RATE)
    segment = burst_iq[start : start + Nw]

    # Time-domain
    axes[0].clear()
    t = np.linspace(0, len(segment) / SAMPLE_RATE, len(segment)) + window_offset
    axes[0].plot(t, segment.real, label="I")
    axes[0].plot(t, segment.imag, label="Q")
    axes[0].set(title="Time-Domain", xlabel="Time [s]", ylabel="I/Q")
    axes[0].legend()

    # Instantaneous freq scatter + bins
    fsamps, bins, out, _ = demodulate_iq(segment, SAMPLE_RATE)
    axes[1].clear()
    axes[1].plot(fsamps, "ro")
    for e in bins:
        axes[1].axhline(e, color="gray", ls="--")
    axes[1].set(title="Inst Freq → Symbols", xlabel="Symbol idx", ylabel="Freq")

    # Eye diagram of full burst IF
    axes[2].clear()
    eye = burst_ifreq.reshape(-1, int(SAMPLE_RATE // DECIMATION // 4800))
    for row in eye:
        axes[2].plot(row, color="tab:blue", alpha=0.3)
    axes[2].set(title="Eye Diagram", xlabel="Sample idx per symbol")

    fig.canvas.draw_idle()


sld_wd.on_changed(update_plot)
sld_wo.on_changed(update_plot)

# ------------------------------------------------------------------------------
# 5) SDR Reader & Burst Detection Thread
# ------------------------------------------------------------------------------
iq_queue = deque()
ringbuf = deque(maxlen=RINGBUF_SIZE)

# envelope filter for burst detection
b_env, a_env = signal.butter(4, ENV_FILT_CUTOFF, fs=SAMPLE_RATE, btype="low")


def reader():
    buff_len = 1 << 14
    buf = np.empty(buff_len, dtype=np.complex64)
    while True:
        r = sdr.readStream(rx_stream, [buf], buff_len, timeoutUs=2_000_000)
        if r.ret > 0:
            chunk = buf[: r.ret].copy()
            iq_queue.append(chunk)
        else:
            time.sleep(0.01)


def detector():
    global burst_iq, burst_ifreq
    noise_floor = None
    burst_active = False
    burst_iq = []
    while True:
        if not iq_queue:
            time.sleep(0.005)
            continue
        chunk = iq_queue.popleft()
        ringbuf.extend(chunk)

        # envelope
        env = np.abs(chunk)
        env = signal.lfilter(b_env, a_env, env)
        mean_env = env.mean()

        if noise_floor is None:
            noise_floor = mean_env
            continue

        if not burst_active and mean_env > noise_floor * THRESH_FACTOR:
            burst_active = True
            burst_iq = list(ringbuf)  # grab preamble

        if burst_active:
            burst_iq.extend(chunk)
            if mean_env < noise_floor * (THRESH_FACTOR * 0.8):
                # burst end
                burst_samples = np.array(burst_iq, dtype=np.complex64)
                # demod full burst (for eye diagram)
                _, _, _, hexstr = demodulate_iq(burst_samples, SAMPLE_RATE)
                print(f"[{datetime.now()}] Decoded burst:", hexstr)

                # prepare data for plotting
                _, _, _, _ = demodulate_iq(burst_samples, SAMPLE_RATE)
                burst_ifreq = demodulate_iq(burst_samples, SAMPLE_RATE)[0]
                burst_iq = burst_samples

                # refresh plots
                update_plot()
                burst_active = False
                burst_iq = []


# ------------------------------------------------------------------------------
# 6) Initialize SDR & Threads
# ------------------------------------------------------------------------------
# SDR init
args = dict(driver="hackrf")
sdr = SoapySDR.Device(args)
sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ)
for g in sdr.listGains(SOAPY_SDR_RX, 0):
    sdr.setGain(SOAPY_SDR_RX, 0, g, 0)
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

# Launch threads
Thread(target=reader, daemon=True).start()
Thread(target=detector, daemon=True).start()

# Start Matplotlib loop
plt.show()
