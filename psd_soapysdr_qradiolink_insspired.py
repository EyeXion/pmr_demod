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


plt.ion()
sample_rate = 4.8e6
# Duration of the recording [s].
D = 0.5
# Number of samples to record.
N = int(D * sample_rate)
center_freq = 433.000e6

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


duration = 0.01  # 20 ms
num_samples = int(sample_rate * duration)  # Number of samples in the window

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

while True:

    index = int(input("Window index ? :"))
    plt.close()
    rx_signal = rx_signal_full[index * num_samples : (index + 1) * num_samples]
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
        N=2,
        Wn=4.8e3,
        fs=sample_rate,
        output="sos",
    )
    rx_signal = signal.sosfilt(sos=low_pass, x=rx_signal)
    # rx_signal = np.convolve(rx_signal, low_pass, "same")
    # We decimate by sample_rate/48khz, hence sample rate should be a multiple of 48k
    # decimation = int(sample_rate / 4.8e3)
    # rx_signal = signal.decimate(rx_signal, q=decimation, ftype="fir")
    # sample_rate = int(sample_rate / decimation)

    # ========================= #
    # Raised Root Cosine Filter #
    # ========================= #

    # ==== #
    # TODO #
    # ==== #

    # =============================== #
    # Instant frequencies calculation #
    # =============================== #

    # The instant frequency is the derivative of the phase
    instant_phases = np.unwrap(np.angle(rx_signal), axis=0)
    instant_frequencies = np.diff(instant_phases) / 2 / np.pi * sample_rate

    samples_per_symbol = int(sample_rate // 4800) * 2
    samples_per_p = int(len(instant_frequencies) // samples_per_symbol)

    time_axis = np.arange(len(instant_frequencies) // samples_per_symbol)

    # =============== #
    # Signal Plotting #
    # =============== #

    # print("Load from {}".format(rx_signal_path))
    # rx_signal = np.load(rx_signal_path)
    print("Signal samples: {}".format(len(rx_signal)))
    print("Signal samples type: {}".format(type(rx_signal[0])))

    # Uncomment this line to plot signal amplitude instead of raw IQ:
    # rx_signal = np.abs(rx_signal)

    plt.figure()
    ax_time = plt.subplot(3, 1, 1)
    t = np.linspace(0, len(rx_signal) / sample_rate, len(rx_signal))
    ax_time.plot(t, rx_signal)
    plt.title("Time-Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude [Complex Number]")

    ax_specgram = plt.subplot(3, 1, 2, sharex=ax_time)
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
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency (Hz)")

    ax_frequency = plt.subplot(3, 1, 3)
    for i in range(0, len(instant_frequencies), samples_per_symbol):
        f = instant_frequencies[i : i + samples_per_symbol]
        time_axis = np.arange(len(f))
        ax_frequency.plot(time_axis, f)
    plt.title("Instanteneous frequencies")
    plt.ylabel("Instantaneous Frequency [Something])")
    plt.xlabel("Sample Index per Symbol")
    plt.show()
