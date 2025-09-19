import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy  # use numpy for buffers

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import numpy as np
import time
from math import log2, ceil, atan2
from pylab import *  # for plotting
from scipy import signal
from threading import Thread

plt.ion()  # turning interactive mode on


# DMR filter - root raised cosine alpha=0.7 Ts = 6650 S/s Fc = 48kHz
xcoeffs = [
    0.0301506278,
    0.0269200615,
    0.0159662432,
    -0.0013114705,
    -0.0216605133,
    -0.0404938748,
    -0.0528141756,
    -0.0543747957,
    -0.0428325003,
    -0.0186176083,
    0.0147202645,
    0.0508418571,
    0.0816392577,
    0.0988113688,
    0.0957187780,
    0.0691512084,
    0.0206194642,
    -0.0431564563,
    -0.1107569268,
    -0.1675773224,
    -0.1981519842,
    -0.1889130786,
    -0.1308939560,
    -0.0218608492,
    0.1325685970,
    0.3190962499,
    0.5182530574,
    0.7070497652,
    0.8623526878,
    0.9644213921,
    1.0000000000,
    0.9644213921,
    0.8623526878,
    0.7070497652,
    0.5182530574,
    0.3190962499,
    0.1325685970,
    -0.0218608492,
    -0.1308939560,
    -0.1889130786,
    -0.1981519842,
    -0.1675773224,
    -0.1107569268,
    -0.0431564563,
    0.0206194642,
    0.0691512084,
    0.0957187780,
    0.0988113688,
    0.0816392577,
    0.0508418571,
    0.0147202645,
    -0.0186176083,
    -0.0428325003,
    -0.0543747957,
    -0.0528141756,
    -0.0404938748,
    -0.0216605133,
    -0.0013114705,
    0.0159662432,
    0.0269200615,
    0.0301506278,
]


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
samples_per_read = int(sample_rate / 9600 / 4)
center_freq = 432.000e6
bandwidth = 100e3
shift_frequency = -1584e3

# Low pass filter parameters
nb_of_taps = 101
cutoff_freq = 24e3
cutoff_freq2 = 6e3  # to smooth out after rrc

low_pass = None
low_pass2 = None
zi = None
zi2 = None


def create_low_pass():
    global low_pass, zi, low_pass2, zi2

    low_pass = signal.butter(
        N=1,
        Wn=cutoff_freq,
        fs=sample_rate,
        output="sos",
    )
    low_pass2 = signal.firwin(
        nb_of_taps,
        cutoff_freq2,
        fs=96000,
    )
    # low_pass2 = signal.butter(N=1, Wn=cutoff_freq2, fs=96000, output="sos")
    # low_pass = signal.cheby1(
    #    N=1, rp=3, Wn=cutoff_freq, btype="lowpass", output="sos", fs=sample_rate
    # )
    # low_pass = signal.cheby2()
    # low_pass = xcoeffs
    # zi = signal.lfilter_zi(low_pass, a=1)
    zi = signal.sosfilt_zi(low_pass)
    # zi2 = signal.sosfilt_zi(low_pass2)
    zi2 = signal.lfilter_zi(low_pass2, a=1)
    fft, f = low_pass_fft()
    # set_subplot(x=f, y=fft, labelx="Fréquence [Hz]", labely="Amplitude", row=1)


def apply_low_pass(samples):
    global zi
    # filtered_samples, zi = signal.lfilter(low_pass, 1, samples, zi=zi)
    # filtered_samples = np.convolve(samples, low_pass, "same")
    filtered_samples, zi = signal.sosfilt(sos=low_pass, x=samples, zi=zi)
    return filtered_samples


def apply_low_pass2(samples):
    global zi2
    # filtered_samples, zi2 = signal.sosfilt(sos=low_pass2, x=samples, zi=zi2)
    filtered_samples = np.convolve(samples, low_pass2, "same")
    return filtered_samples


def apply_raised_cosine(samples):
    filtered_samples = np.convolve(samples, xcoeffs, "same")
    return filtered_samples


def low_pass_fft():
    # plot the frequency response
    """
    fft = np.abs(np.fft.fft(low_pass, 1024))  # take the 1024-point FFT and magnitude
    fft = np.fft.fftshift(fft)  # make 0 Hz in the center
    f = np.linspace(
        -sample_rate / 2, sample_rate / 2, len(fft)
    )  # x axis plt.plot(w, H, '.-')
    return fft, f
    """

    w, h = signal.sosfreqz(low_pass, fs=sample_rate)
    # w, h = signal.freqz(low_pass, a=1, fs=sample_rate)
    return h, w


def set_subplot(
    x, y, labelx, labely, row=None, col=None, marker="-", xlim=None, ylim=None
):

    if row is None and col is None:
        figure = ax
    else:
        if row is None:
            figure = ax[col]
        elif col is None:
            figure = ax[row]
        else:
            figure = ax[row, col]

    figure.clear()

    if xlim is not None:
        figure.set_xlim(xlim)
    if ylim is not None:
        figure.set_ylim(ylim)

    figure.plot(x, y, marker)
    figure.set_xlabel(labelx)
    figure.set_ylabel(labely)
    plt.plot()


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


fig, ax = plt.subplots(nrows=3)
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
    valmin=32,
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


def filter_samples(buff):
    """
    Return the filtered samples and new sample rate after decimation
    """
    N = len(buff)  # number of samples
    samples = buff * np.hamming(N)  # appliquer une fenêtre de Hamming
    final_decimation = 1

    t = np.arange(len(buff)) / sample_rate
    # Filtre hétérodyne comme en SDR pour shift la fréquence
    mixing_signal = np.exp(1j * 2 * np.pi * shift_frequency * t)
    shifted_signal = samples * mixing_signal

    # Apply low pass filter
    filtered_samples = apply_low_pass(shifted_signal)

    # Decimate to onlykeep what we need
    # To be safe, we multiply the bandwidth by 1.5 for margin
    decimation_factor = int(ceil(sample_rate / (cutoff_freq * 2)))
    # filtered_samples = filtered_samples[::decimation_factor]
    filtered_samples = signal.decimate(
        filtered_samples, q=decimation_factor, ftype="fir"
    )
    final_decimation *= decimation_factor

    # apply raised cosine filter
    filtered_samples = apply_raised_cosine(filtered_samples)
    # decimation_factor_rrc = int(sample_rate / decimation_factor) / (48e3 * 2)
    # filtered_samples = filtered_samples[::decimation_factor_rrc]
    # filtered_samples = signal.resample_poly(
    #    filtered_samples, up=int(sample_rate / decimation_factor), down=cutoff_freq
    # )
    """
    filtered_samples = signal.decimate(
        filtered_samples, q=decimation_factor_rrc, ftype="fir"
    )
    final_decimation *= decimation_factor_rrc
    """

    # Apply the last filter to remove noise before FSK demod
    # Exit of last filter is always a 48Khz bandwidth -> 96000 sample rate
    # here we end up with a 6kHz sample rate -> decimation by 16
    filtered_samples = apply_low_pass2(filtered_samples)
    decimation_factor_last = int(int(sample_rate / (final_decimation)) / 12e3)
    filtered_samples = signal.decimate(
        filtered_samples, q=decimation_factor_last, ftype="fir"
    )
    final_decimation *= decimation_factor_last
    print(sample_rate / final_decimation)

    return filtered_samples, int(sample_rate / final_decimation)


def calculate_psd(filtered_samples, t, Fs):

    N = len(filtered_samples)  # number of samples
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


# Regarder https://en.wikipedia.org/wiki/Instantaneous_phase_and_frequency
# ou https://pp4fpgas.readthedocs.io/en/latest/project7.html
# or https://dsp.stackexchange.com/questions/24487/calculate-and-interpret-the-instantaneous-frequency
def get_instant_frequencies(filtered_samples, sample_rate):
    frequencies = []

    """
    x = filtered_samples[1:]
    y = np.conjugate(filtered_samples[:-1])
    conj = np.multiply(x, y)
    factor = 6.28 * sample_rate
    for z in conj:
        f = atan2(z.real, z.imag) * factor
        frequencies.append(f)
    """

    re = np.real(filtered_samples)
    im = np.imag(filtered_samples)
    for i in range(1, len(filtered_samples)):
        # using Barns estimation of instantaneous frequency
        # Since we have a sampled array, no need for Fs
        # formula 11
        a = re[i - 1] * im[i]
        b = re[i] * im[i - 1]
        c = (re[i - 1] + re[i]) ** 2
        d = (im[i - 1] + im[i]) ** 2

        f = 2 / 3.14 * atan2(a - b, c + d)
        frequencies.append(f)

    frequencies = np.array(frequencies, np.float64)
    n = np.arange(0, len(frequencies))
    return frequencies, n


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

sample_cumulation = np.array([], np.complex64)

frequencies_cumulation = np.array([])

create_low_pass()
while True:
    # temporal plot
    buff = total_buff.copy()
    filtered_samples, decimated_sample_rate = filter_samples(buff)
    t = np.arange(len(filtered_samples)) / decimated_sample_rate

    sample_cumulation = np.concatenate((sample_cumulation, filtered_samples))[
        -len(filtered_samples) * 20 :
    ]
    set_subplot(
        x=np.real(sample_cumulation),
        y=np.imag(sample_cumulation),
        labelx="Re",
        labely="Im",
        row=1,
        marker="ro",
        xlim=[-1.0, 1.0],
        ylim=[-1.0, 1.0],
    )
    frequencies, n = get_instant_frequencies(
        filtered_samples, len(filtered_samples / decimated_sample_rate)
    )
    frequencies_cumulation = np.append(
        frequencies_cumulation, np.mean(frequencies)[-200:]
    )
    set_subplot(
        x=np.arange(0, len(frequencies_cumulation), 1),
        y=frequencies_cumulation,
        labelx="Sample number",
        labely="Instant frequency",
        row=2,
    )
    psd, f = calculate_psd(filtered_samples, t, decimated_sample_rate)
    # f, psd = signal.welch(filtered_samples, fs=decimated_sample_rate, return_onesided=False)
    set_subplot(
        x=f,
        y=psd,
        labelx="Fréquence [Hz]",
        labely="Amplitude [dB]",
        row=0,
        ylim=[-150, 0],
    )
    plt.grid(True)
    plt.pause(0.01)


# shutdown the stream
sdr.deactivateStream(rxStream)  # stop streaming
sdr.closeStream(rxStream)
