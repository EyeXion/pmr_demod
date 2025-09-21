import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy  # use numpy for buffers

import matplotlib.pyplot as plt

from matplotlib import mlab
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import time
from math import log2, ceil, atan2
from pylab import *  # for plotting
from scipy import signal
from threading import Thread
from fractions import Fraction
from datetime import datetime


# From https://github.com/veeresht/CommPy/blob/master/commpy/filters.py
def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.

    Parameters
    ----------
    N : int
        Length of the filter in samples.

    alpha : float
        Roll off factor (Valid values are [0, 1]).

    Ts : float
        Symbol period in seconds.

    Fs : float
        Sampling Rate in Hz.

    Returns
    ---------

    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.

    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1 / float(Fs)
    time_idx = ((np.arange(N) - N / 2)) * T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
            )
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
            )
        else:
            h_rrc[x] = (
                np.sin(np.pi * t * (1 - alpha) / Ts)
                + 4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)
            ) / (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

    return time_idx, h_rrc


sample_rate = 2.4e6
# Duration of the recording [s].
D = 0.5
# Number of samples to record.
N = int(D * sample_rate)
center_freq = 433.000e6

# Decimation value after first low pass filter
decimation = int(sample_rate / 48e3)


# RRC filter (shape filter)
rrc_filter = rrcosfilter(501, 0.2, 1 / 4800, int(sample_rate / decimation))[1]

# enumerate devices
results = SoapySDR.Device.enumerate()
for result in results:
    print(result)

# create device instance
# args can be user defined or from the enumeration result
args = dict(driver="hackrf")
sdr = SoapySDR.Device(args)

# apply settings
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
print(sdr.listGains(SOAPY_SDR_RX, 0))
sdr.setGain(SOAPY_SDR_RX, 0, "AMP", 0)
sdr.setGain(SOAPY_SDR_RX, 0, "LNA", 0)
sdr.setGain(SOAPY_SDR_RX, 0, "VGA", 0)

rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

# Create a re-usable buffer for RX samples.
# - The optimal buffer size is a user choice depending on the used SDR and the sampling rate.
# - If the buffer is too large, it will consume all memory.
# - If the buffer is too small, it will produces overflow as the computer will not consume fast enough the SDR's samples.
# - The highest buffer size supported by the HackRF is 2^17.
# - To use the USRP with a sampling rate of 30e6 recording during 1 seconnd, use at least a buffer size of 2^24.
rx_buff_len = pow(2, 17)
rx_buff = np.array([0] * rx_buff_len, np.complex64)
# Create the buffer for the final recorded signal.
rx_signal = np.array([0], np.complex64)

# Until the desired number of samples is received.
while len(rx_signal) < N:
    # Increase timeout to 10s as some SDRs can be slow to send samples.
    sr = sdr.readStream(rx_stream, [rx_buff], rx_buff_len, timeoutUs=10000)
    # Check the readStream operation.
    # The following line seems to accept overflows, hence, samples set to 0:
    # if sr.ret > 0 and sr.flags == 1 << 2:
    if sr.ret == rx_buff_len:
        print("Number of samples: sr.ret: {}".format(sr.ret))
        print("Timestamp for receive buffer: sr.timeNs: {}".format(sr.timeNs))
        rx_signal = np.concatenate((rx_signal, rx_buff))
    else:
        print(
            "Error code: sr.ret: {}".format(sr.ret)
        )  # See "/usr/local/include/SoapySDR/Errors.h".
        print(
            "Flags set by receive operation: sr.flags: {0:b}b".format(sr.flags)
        )  # See "/usr/local/include/SoapySDR/Constants.h".
# Truncate to desired number of samples.
rx_signal = rx_signal[:N]

# Stop streaming and close stream.
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

date = datetime.now().strftime("%Y%m%d_%H_%M_%S%m")
rx_signal_path = "DMRCapture_HackRF_" + str(int(sample_rate)) + "Hz_" + date + ".npy"

print("Signal samples: {}".format(len(rx_signal)))
print("Save to {}".format(rx_signal_path))
np.save(rx_signal_path, rx_signal)


duration = 0.015  # 20 ms

# ====== #
# Mixing #
# ====== #
t = np.arange(len(rx_signal)) / sample_rate
shift_frequency = (
    center_freq - 433.582e6
)  # If talkie frequency is 433.575Mhz, center of signal is at 433.582Mhz
mixing_signal = np.exp(1j * 2 * np.pi * shift_frequency * t)
rx_signal = rx_signal * mixing_signal


rx_signal_full = rx_signal.copy()
original_sample_rate = sample_rate

window_duration = 0.015
window_offset = 0

fig, ax = plt.subplots(nrows=3)
fig.subplots_adjust(left=0.15, right=0.85)
axwindow_duration = fig.add_axes([0.05, 0.25, 0.01, 0.65])
window_duration_slider = Slider(
    ax=axwindow_duration,
    label="Window duration [s]",
    valmin=0.001,
    valmax=D - window_offset,
    valstep=0.001,
    valinit=window_duration,
    orientation="vertical",
)

axwindow_offset = fig.add_axes([0.95, 0.25, 0.01, 0.65])
window_offset_slider = Slider(
    ax=axwindow_offset,
    label="Window offset [s]",
    valmin=0,
    valmax=D - window_duration,
    valstep=0.001,
    valinit=window_offset,
    orientation="vertical",
)


def update(val):

    global window_duration, window_offset
    window_duration = window_duration_slider.val

    window_offset = min(window_offset_slider.val, D - window_duration)
    window_offset_slider.valmax = D - window_duration
    window_offset_slider.ax.set_xlim(0, D - window_duration)

    sample_rate = original_sample_rate
    num_samples = int(sample_rate * window_duration)  # Number of samples in the window
    start_sample_index = int(sample_rate * window_offset)

    rx_signal = rx_signal_full[start_sample_index : start_sample_index + num_samples]

    # =============== #
    # Low Pass Filter #
    # =============== #

    # We need to apply a low pass filter in order to have a 48kHz sample rate
    nb_taps = 101
    """
    low_pass = signal.firwin(
        nb_taps,
        4.8e3,
        fs=sample_rate,
    )
    """
    low_pass = signal.butter(
        N=15,
        Wn=9.6e3,
        fs=sample_rate,
        output="sos",
    )
    """
    low_pass = signal.cheby1(
        N=5, rp=2, Wn=9.6e3, btype="lowpass", output="sos", fs=sample_rate
    )
    """

    rx_signal = signal.sosfilt(sos=low_pass, x=rx_signal)
    # rx_signal = np.convolve(rx_signal, low_pass, "same")
    # We decimate by sample_rate/48khz, hence sample rate should be a multiple of 48k
    rx_signal = signal.decimate(rx_signal, q=decimation, ftype="fir")
    sample_rate = int(sample_rate / decimation)
    print(sample_rate)

    # ==== #
    # TODO #
    # ==== #

    # =============================== #
    # Instant frequencies calculation #
    # =============================== #

    # The instant frequency is the derivative of the phase

    samples_per_symbol = int(sample_rate // 4800) * 2
    instant_phases = np.unwrap(np.angle(rx_signal), axis=0)
    deviation = 1944
    instant_frequencies = np.diff(instant_phases) * (
        samples_per_symbol / (2 * np.pi * deviation)
    )

    time_axis = np.arange(len(instant_frequencies) // samples_per_symbol)

    # ========================= #
    # Raised Root Cosine Filter #
    # ========================= #
    instant_frequencies = np.convolve(rrc_filter, instant_frequencies, "same")

    # =============== #
    # Signal Plotting #
    # =============== #

    # print("Load from {}".format(rx_signal_path))
    # rx_signal = np.load(rx_signal_path)
    print("Signal samples: {}".format(len(rx_signal)))
    print("Signal samples type: {}".format(type(rx_signal[0])))

    # Uncomment this line to plot signal amplitude instead of raw IQ:
    # rx_signal = np.abs(rx_signal)

    ax_time = ax[0]
    ax_time.clear()
    t = np.linspace(0, len(rx_signal) / sample_rate, len(rx_signal))
    ax_time.plot(t, rx_signal)
    ax_time.set_title("Time-Domain")
    ax_time.set_xlabel("Time [s]")
    ax_time.set_ylabel("Magnitude [Complex Number]")

    """
    ax_specgram = ax[1]
    ax_specgram.clear()
    ax_specgram.sharex(ax_time)
    ax_specgram.specgram(
        rx_signal,
        NFFT=256,
        Fs=sample_rate,
        Fc=0,
        detrend=mlab.detrend_none,
        window=mlab.window_hanning,
        noverlap=127,
        cmap=None,
        xextent=None,
        pad_to=None,
        sides="default",
        scale_by_freq=None,
        mode="default",
        scale="default",
    )
    ax_specgram.set_title("Spectrogram")
    ax_specgram.set_xlabel("Time [s]")
    ax_specgram.set_ylabel("Frequency (Hz)")
    """

    ax_iq = ax[1]
    ax_iq.clear()
    ax_iq.plot(np.real(rx_signal), np.imag(rx_signal), "x")
    ax_iq.set_title("Complex Plan")
    ax_iq.set_xlabel("Re")
    ax_iq.set_ylabel("Im")

    ax_frequency = ax[2]
    ax_frequency.clear()
    for i in range(0, len(instant_frequencies), samples_per_symbol):
        f = instant_frequencies[i : i + samples_per_symbol]
        time_axis = np.arange(len(f))
        ax_frequency.plot(time_axis, f)
    ax_frequency.set_title("Instanteneous frequencies")
    ax_frequency.set_ylabel("Instantaneous Frequency [Something])")
    ax_frequency.set_xlabel("Sample Index per Symbol")
    fig.canvas.draw_idle()


window_duration_slider.on_changed(update)
window_offset_slider.on_changed(update)
update(0)
plt.show()
