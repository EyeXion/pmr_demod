import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy  # use numpy for buffers

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import time
from math import log2, ceil
from pylab import *  # for plotting
from scipy import signal
from threading import Thread

plt.ion()  # turning interactive mode on


def get_nb_samples_per_read(sample_rate, fft_resolution):
    exact_samples_per_read = int(sample_rate / fft_resolution)

    # closest power of 2 (higher or equal)
    power = log2(exact_samples_per_read)
    if isinstance(power, int):
        return int(2**power)
    else:
        return int(2 ** (ceil(power)))


sample_rate = 8e6
samples_per_read = int(
    get_nb_samples_per_read(sample_rate=sample_rate, fft_resolution=1000) / 2
)
center_freq = 432.000e6
bandwidth = 100e3
shift_frequency = -1575e3

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

# setup a stream (complex floats)
rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream)  # start streaming

# Concatenate buffer of 4 captures max
total_buff = np.array([], np.complex64)

# create a re-usable buffer for rx samples
rx_buff = np.array([0] * samples_per_read, np.complex64)


fig, ax = plt.subplots()
fig.subplots_adjust(left=0.25, bottom=0.25)
axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
center_freq_slider = Slider(
    ax=axfreq,
    label="Frequency [MHz]",
    valmin=430,
    valmax=470,
    valstep=0.1,
    valinit=center_freq * 10**-6,
)
axshift = fig.add_axes([0.25, 0.15, 0.65, 0.03])
shift_frequency_slider = Slider(
    ax=axshift,
    label="Shift Frequency (mixing) [KHz]",
    valmin=-3000,
    valmax=3000,
    valstep=1,
    valinit=shift_frequency * 10**-3,
)

axsamplerate = fig.add_axes([0.1, 0.25, 0.03, 0.65])
sample_rate_slider = Slider(
    ax=axsamplerate,
    label="Sample Rate [Mb/s]",
    valmin=8,
    valmax=20,
    valstep=1,
    valinit=sample_rate * 10**-6,
    orientation="vertical",
)


def calculate_psd():
    buff = total_buff.copy()

    Fs = sample_rate  # sample rate
    Ts = 1 / Fs  # sample period
    N = len(buff)  # number of samples

    samples = buff * np.hamming(len(total_buff))  # appliquer une fenêtre de Hamming
    t = np.arange(len(samples)) / sample_rate
    # Filtre hétérodyne comme en SDR pour shift la fréquence
    mixing_signal = np.exp(1j * 2 * np.pi * shift_frequency * t)
    shifted_signal = samples * mixing_signal

    # PSD, donc puissance par fréquence de la FFT, donc le carré de la magnitude
    # Puis normalisation
    PSD = np.abs(np.fft.fft(shifted_signal)) ** 2 / (N * Fs)  # Calcul de la PSD
    PSD_log = 10.0 * np.log10(PSD)  # Utiliser échelle en log
    PSD_shifted = np.fft.fftshift(PSD_log)  # Mise en bande de base

    # tableau des fréquences de la FFT (fftfreq en gros)
    f = np.arange(Fs / -2.0, Fs / 2.0, Fs / N)  # début, fin, pas, centré autour de 0 Hz
    f += center_freq - shift_frequency  # Le centre est la fréquence de la SDR

    return PSD_shifted, f


def sample_iq():
    global total_buff
    while True:
        sr = sdr.readStream(rxStream, [rx_buff], len(rx_buff))
        total_buff = np.concatenate((total_buff[-samples_per_read:], rx_buff[: sr.ret]))


def update_center_freq(val):
    global center_freq
    center_freq = center_freq_slider.val * 10**6
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)


def update_shift_frequency(val):
    global shift_frequency
    shift_frequency = shift_frequency_slider.val * 10**3


def update_sample_rate(val):
    global sample_rate
    sample_rate = sample_rate_slider.val * 10**6
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)


center_freq_slider.on_changed(update_center_freq)
shift_frequency_slider.on_changed(update_shift_frequency)
sample_rate_slider.on_changed(update_sample_rate)

sampling_thread = Thread(target=sample_iq)
sampling_thread.start()

time.sleep(1)

while True:
    psd, f = calculate_psd()
    ax.clear()
    ax.plot(f, psd)
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid(True)
    plt.pause(0.1)


# shutdown the stream
sdr.deactivateStream(rxStream)  # stop streaming
sdr.closeStream(rxStream)
