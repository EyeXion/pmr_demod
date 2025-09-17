from libhackrf import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import time
from ctypes import c_int, CFUNCTYPE, POINTER
from pylab import *  # for plotting
from threading import Thread
from scipy import signal

recording_time = 0.2  # seconds
sample_rate = 10e6
center_freq = 432.000e6
bandwidth = 100e3
shift_frequency = -1575e3

hackrf = HackRF()

hackrf.sample_rate = sample_rate
hackrf.center_freq = center_freq
# hackrf.bandwidth = bandwidth


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


def calculate_psd(samples):
    Fs = sample_rate  # sample rate
    Ts = 1 / Fs  # sample period
    N = len(samples)  # number of samples

    # PSD, donc puissance par fréquence de la FFT, donc le carré de la magnitude
    # Puis normalisation
    PSD = np.abs(np.fft.fft(samples)) ** 2 / (N * Fs)  # Calcul de la PSD
    PSD_log = 10.0 * np.log10(PSD)  # Utiliser échelle en log
    PSD_shifted = np.fft.fftshift(PSD_log)  # Mise en bande de base

    # tableau des fréquences de la FFT (fftfreq en gros)
    f = np.arange(Fs / -2.0, Fs / 2.0, Fs / N)  # début, fin, pas, centré autour de 0 Hz
    f += center_freq - shift_frequency  # Le centre est la fréquence de la SDR

    return PSD_shifted, f


def sample_iq():
    samples = hackrf.read_samples(sleep_time=recording_time)

    samples = samples[
        100000:
    ]  # get rid of the first 100k samples just to be safe, due to transients

    samples = samples * np.hamming(len(samples))  # appliquer une fenêtre de Hamming

    t = np.arange(len(samples)) / sample_rate
    # Filtre hétérodyne comme en SDR pour shift la fréquence
    mixing_signal = np.exp(1j * 2 * np.pi * shift_frequency * t)
    shifted_signal = samples * mixing_signal

    return shifted_signal


def rx_callback(
    hackrf_transfer,
):  # this callback function always needs to have these four args
    global samples, last_idx
    valid_length = hackrf_transfer.valid_length
    buffer = hackrf_transfer.buffer

    accepted = valid_length // 2
    accepted_samples = buffer[:valid_length].astype(np.int8)  # -128 to 127
    accepted_samples = (
        accepted_samples[0::2] + 1j * accepted_samples[1::2]
    )  # Convert to complex type (de-interleave the IQ)
    accepted_samples /= 128  # -1 to +1
    samples[last_idx : last_idx + accepted] = accepted_samples

    last_idx += accepted

    return 0


shifted_signal = None


def sample_acquisition_loop():
    global shifted_signal
    while True:
        shifted_signal = sample_iq()


def update_plot(frame):
    psd, f = calculate_psd(shifted_signal)
    ax.clear()
    ax.plot(f, psd)
    plt.xlabel("Fréquence [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid(True)
    fig.canvas.draw()


def update_center_freq(val):
    global center_freq
    center_freq = center_freq_slider.val * 10**6
    hackrf.center_freq = center_freq


def update_shift_frequency(val):
    global shift_frequency
    shift_frequency = shift_frequency_slider.val * 10**3


def update_sample_rate(val):
    global sample_rate
    sample_rate = sample_rate_slider.val * 10**6
    hackrf.sample_rate = sample_rate


center_freq_slider.on_changed(update_center_freq)
shift_frequency_slider.on_changed(update_shift_frequency)
sample_rate_slider.on_changed(update_sample_rate)


sample_aquisition_thread = Thread(target=sample_acquisition_loop)
sample_aquisition_thread.start()

while shifted_signal is None:
    pass

anim = FuncAnimation(fig, update_plot, interval=recording_time * 1000)
plt.show()
