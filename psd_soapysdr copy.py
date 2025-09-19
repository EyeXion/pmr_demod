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
samples_per_read = get_nb_samples_per_read(sample_rate=sample_rate, fft_resolution=500)
center_freq = 432.000e6
bandwidth = 100e3
shift_frequency = -1575e3

# Low pass filter parameters
nb_of_taps = 101
cutoff_freq = 15e3  # 20khZ bandwidth for the channel, 12.5 kHz but margin

low_pass = None
zi = None


def create_low_pass():
    global low_pass, zi
    """
    low_pass = signal.firwin(
        nb_of_taps,
        cutoff_freq/2,
        fs=sample_rate,
        width=transition_width,
    )
    """

    """
    low_pass = signal.butter(
        N=1,
        Wn=cutoff_freq,
        fs=sample_rate,
        output="sos",
    )
    """
    low_pass = signal.cheby1(
        N=1, rp=3, Wn=cutoff_freq, btype="lowpass", output="sos", fs=sample_rate
    )
    # low_pass = signal.cheby2()
    # zi = signal.lfilter_zi(low_pass)
    zi = signal.sosfilt_zi(low_pass)
    """
    ax[1].clear()
    ax[1].plot(f, fft)
    ax[1].set_xlabel("Fréquence [Hz]")
    ax[1].set_ylabel("Amplitude")
    """
    fft, f = low_pass_fft()
    set_subplot(x=f, y=fft, labelx="Fréquence [Hz]", labely="Amplitude", row=1)


def apply_low_pass(samples):
    global zi
    # filtered_samples, zi = signal.lfilter(low_pass, 1, samples, zi=zi)
    # filtered_samples = np.convolve(samples, low_pass, "same")
    filtered_samples, zi = signal.sosfilt(sos=low_pass, x=samples, zi=zi)
    return filtered_samples


def low_pass_fft():
    """
    # plot the frequency response
    fft = np.abs(np.fft.fft(low_pass, 1024))  # take the 1024-point FFT and magnitude
    fft = np.fft.fftshift(fft)  # make 0 Hz in the center
    f = np.linspace(
        -sample_rate / 2, sample_rate / 2, len(fft)
    )  # x axis plt.plot(w, H, '.-')
    """
    w, h = signal.sosfreqz(low_pass, fs=sample_rate)
    return h, w


def set_subplot(x, y, labelx, labely, row=None, col=None):

    if row is None and col is None:
        figure = ax
    else:
        if row is None:
            figure = ax[col]
        elif col is None:
            figure = ax[row]
        else:
            figure = ax[row, col]

    print("coucou")
    figure.clear()
    figure.plot(x, y)
    figure.set_xlabel(labelx)
    figure.set_ylabel(labely)


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


fig, ax = plt.subplots(nrows=2)
fig.subplots_adjust(left=0.10, bottom=0.30, right=0.90)
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
axsamplerate = fig.add_axes([0.25, 0.2, 0.65, 0.03])
sample_rate_slider = Slider(
    ax=axsamplerate,
    label="Sample Rate [Mb/s]",
    valmin=8,
    valmax=20,
    valstep=1,
    valinit=sample_rate * 10**-6,
)


axcutoff = fig.add_axes([0.05, 0.25, 0.03, 0.65])
cutoff_freq_slider = Slider(
    ax=axcutoff,
    label="LowPass Cutoff Freq [kHz]",
    valmin=1,
    valmax=100,
    valstep=0.5,
    valinit=cutoff_freq * 10**-3,
    orientation="vertical",
)

"""
axtransition = fig.add_axes([0.95, 0.25, 0.03, 0.65])
transition_width_slider = Slider(
    ax=axtransition,
    label="LowPass Transition Width [kHz]",
    valmin=1,
    valmax=1000,
    valstep=0.5,
    valinit=transition_width * 10**-3,
    orientation="vertical",
)
"""


def calculate_psd():
    buff = total_buff.copy()

    N = len(buff)  # number of samples
    samples = buff * np.hamming(N)  # appliquer une fenêtre de Hamming

    t = np.arange(len(samples)) / sample_rate
    # Filtre hétérodyne comme en SDR pour shift la fréquence
    mixing_signal = np.exp(1j * 2 * np.pi * shift_frequency * t)
    shifted_signal = samples * mixing_signal

    # Apply low pass filter
    filtered_samples = apply_low_pass(shifted_signal)

    # Decimate to onlykeep what we need
    # To be safe, we multiply the bandwidth by 1.5 for margin
    decimation_factor = int(sample_rate / (cutoff_freq * 2 * 1.5))
    filtered_samples = filtered_samples[::decimation_factor]

    Fs = int(sample_rate / decimation_factor)  # sample rate

    # PSD, donc puissance par fréquence de la FFT, donc le carré de la magnitude
    # Puis normalisation
    PSD = np.abs(np.fft.fft(filtered_samples)) ** 2 / (N * Fs)  # Calcul de la PSD
    PSD_log = 10.0 * np.log10(PSD)  # Utiliser échelle en log
    PSD_shifted = np.fft.fftshift(PSD_log)  # Recentrer DC comme IQ sampling

    # tableau des fréquences de la FFT (fftfreq en gros)
    f = np.arange(
        Fs / -2.0, Fs / 2.0, Fs / len(filtered_samples)
    )  # début, fin, pas, centré autour de 0 Hz
    f += center_freq - shift_frequency  # Le centre est la fréquence de la SDR

    return PSD_shifted, f


def sample_iq():
    global total_buff
    while True:
        sr = sdr.readStream(rxStream, [rx_buff], len(rx_buff))
        # total_buff = np.concatenate((total_buff[-samples_per_read:], rx_buff[: sr.ret]))
        total_buff = rx_buff


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
    create_low_pass()


def update_cutoff(val):
    global cutoff_freq
    cutoff_freq = cutoff_freq_slider.val * 10**3
    create_low_pass()


"""
def update_transition(val):
    global transition_width
    transition_width = transition_width_slider.val * 10**3
    create_low_pass()
"""

center_freq_slider.on_changed(update_center_freq)
shift_frequency_slider.on_changed(update_shift_frequency)
sample_rate_slider.on_changed(update_sample_rate)
cutoff_freq_slider.on_changed(update_cutoff)
# transition_width_slider.on_changed(update_transition)

sampling_thread = Thread(target=sample_iq)
sampling_thread.start()

time.sleep(1)

create_low_pass()
while True:
    psd, f = calculate_psd()
    set_subplot(x=f, y=psd, labelx="Fréquence [Hz]", labely="Amplitude [dB]", row=0)

# shutdown the stream
sdr.deactivateStream(rxStream)  # stop streaming
sdr.closeStream(rxStream)
