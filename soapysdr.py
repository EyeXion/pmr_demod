#!/usr/bin/python3

# Snippet of SoapySDR usage to initialize an SDR and record a signal.
# Supports for USRP and HackRF.
# It uses Numpy and Matplotlib to plot the recorded signal.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import mlab
import SoapySDR

# Sample rate [samples / s], hence defining the bandwidth.
fs = 2e6
# Center frequency [Hz].
freq = 433.000e6
# Duration of the recording [s].
D = 3
# Number of samples to record.
N = int(D * fs)
# Signal file (optionnal, everything can be done in memory).
rx_signal_path = "/tmp/rx_signal.npy"

print("Sample rate: fs={}".format(fs))
print("Center frequency: freq={}".format(freq))
print("Duration: D={}".format(D))
print("Number of samples: N={}".format(N))

if __name__ == "__main__":
    # ================== #
    # SDR Initialization #
    # ================== #
    
    # Check for an available SDR and create device instance.
    results = SoapySDR.Device.enumerate()
    for result in results: print(result)
    if not results:
        print("SoapySDR doesn't detect any available SDR!")
        exit(1)

    args = dict(driver="hackrf")
    sdr = SoapySDR.Device(args)
    
    # Query device info.
    print("sdr.listAntennas:       {}".format(sdr.listAntennas(SoapySDR.SOAPY_SDR_RX, 0)))
    print("sdr.listGains:          {}".format(sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)))
    freqRanges = sdr.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
    for freqRange in freqRanges:
        print("sdr.getFrenquencyRange: {}".format(freqRange))
    
    # Apply paramters.
    sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, fs)
    sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
    # Set specific gain depending on the used SDR.
    if "HackRF" in str(sdr):
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'AMP', 14) # [dB] 100%.
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'LNA', 40) # [dB] 100%.
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 'VGA', 22) # [dB] 30%.
    if "b200" in str(sdr):
        sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, 76)        # [dB] 100%.
        sdr.setAntenna(SoapySDR.SOAPY_SDR_RX, 0, "TX/RX")
        
    # Check parameters.
    print("sdr.getSampleRate:      {}".format(sdr.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)))
    print("sdr.getFrequency:       {}".format(sdr.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)))
    print("sdr.getGain:            {}".format(sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0)))
    print("sdr.getGainMode:        {}".format(sdr.getGainMode(SoapySDR.SOAPY_SDR_RX, 0)))
    print("sdr.getAntenna:         {}".format(sdr.getAntenna(SoapySDR.SOAPY_SDR_RX, 0)))
    if "HackRF" in str(sdr):
        print("sdr.getGain(RF):        {}".format(sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, 'LNA')))
        print("sdr.getGain(IF):        {}".format(sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, 'AMP')))
        print("sdr.getGain(BB):        {}".format(sdr.getGain(SoapySDR.SOAPY_SDR_RX, 0, 'VGA')))

    # ================ #
    # Signal Recording #
    # ================ #
        
    # Setup stream in complex float and start streaming.
    # Note that 'CF32' in SoapySDR equal 'complex64' in Numpy (same for 'CF64' and 'complex128').
    # `{setup,activate}Stream` must always be associated to one single `{close,deactivate}Stream` call.
    # Otherwise, an end-of-burst error will be thrown.
    rx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
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
        sr = sdr.readStream(rx_stream, [rx_buff], rx_buff_len, timeoutUs=10000000)
        # Check the readStream operation.
        # The following line seems to accept overflows, hence, samples set to 0:
        # if sr.ret > 0 and sr.flags == 1 << 2:
        if sr.ret == rx_buff_len and sr.flags == 1 << 2:
            print("Number of samples: sr.ret: {}".format(sr.ret))
            print("Timestamp for receive buffer: sr.timeNs: {}".format(sr.timeNs))
            rx_signal = np.concatenate((rx_signal, rx_buff))
        else:
            print("Error code: sr.ret: {}".format(sr.ret))                             # See "/usr/local/include/SoapySDR/Errors.h".
            print("Flags set by receive operation: sr.flags: {0:b}b".format(sr.flags)) # See "/usr/local/include/SoapySDR/Constants.h".
    # Truncate to desired number of samples.
    rx_signal = rx_signal[:N]

    # Stop streaming and close stream.
    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    # ============= #
    # Signal Saving #
    # ============= #

    print("Signal samples: {}".format(len(rx_signal)))
    print("Save to {}".format(rx_signal_path))
    np.save(rx_signal_path, rx_signal)
    
    # =============== #
    # Signal Plotting #
    # =============== #

    print("Load from {}".format(rx_signal_path))
    rx_signal = np.load(rx_signal_path)
    print("Signal samples: {}".format(len(rx_signal)))
    print("Signal samples type: {}".format(type(rx_signal[0])))

    # Uncomment this line to plot signal amplitude instead of raw IQ:
    # rx_signal = np.abs(rx_signal)

    plt.figure()
    ax_time = plt.subplot(2, 1, 1)
    t = np.linspace(0, len(rx_signal) / fs, len(rx_signal))
    ax_time.plot(t, rx_signal)
    plt.title("Time-Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude [Complex Number]")
    
    ax_specgram = plt.subplot(2, 1, 2, sharex=ax_time)
    ax_specgram.specgram(rx_signal, NFFT=256, Fs=fs, Fc=0,
                 detrend=mlab.detrend_none, window=mlab.window_hanning,
                 noverlap=127, cmap=None, xextent=None, pad_to=None,
                 sides='default', scale_by_freq=None, mode='default', scale='default')
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency (Hz)")
    plt.show()
