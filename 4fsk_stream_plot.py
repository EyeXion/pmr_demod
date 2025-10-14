import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
from scipy import signal
import time
import matplotlib.pyplot as plt
from math import ceil

# ================= #
# Demodulator Class #
# ================= #
# SYMBOLS = np.array([0, 3, 1, -1, -3], dtype=int)
SYMBOLS = np.array([0, -3, -1, 1, 3], dtype=np.int32)


_SYMBOL_TO_BITS = {
    3: "01",
    1: "00",
    -1: "10",
    -3: "11",
}

SYNC_PATTERNS = {
    "BS_SOURCED_VOICE": "755fd7df75f7",
    "BS_SOURCED_DATA": "dff57d75df5d",
    "MS_SOURCED_VOICE": "7f7d5dd57dfd",
    "MS_SOURCED_DATA": "d5d7f77fd757",
    "MS_SOURCED_RC": "77d55f7dfd77",
    "TDMA_DIRECT_S1_VOICE": "5d577f7757ff",
    "TDMA_DIRECT_S1_DATA": "f7fdd5ddfd55",
    "TDMA_DIRECT_S2_VOICE": "7dffd5f55d5f",
    "TDMA_DIRECT_D2_DATA": "d7557f5ff7f5",
}


class DMRStreamDemodulator:
    """
    A class to continuously demodulate 4FSK DMR bursts from an SDR stream,
    with a maximum duration per burst and optional plotting.
    """

    # RRC filter taps (dunno how they calculated it, but it works for 48kHz)
    RRCOS_FILTER = [
        +0.0273676736,
        +0.0190682959,
        +0.0070661879,
        -0.0075385898,
        -0.0231737159,
        -0.0379433607,
        -0.0498333862,
        -0.0569528373,
        -0.0577853377,
        -0.0514204905,
        -0.0377352004,
        -0.0174982391,
        +0.0076217868,
        +0.0351552125,
        +0.0620353691,
        +0.0848941519,
        +0.1004237235,
        +0.1057694293,
        +0.0989127431,
        +0.0790009892,
        +0.0465831968,
        +0.0037187043,
        -0.0460635022,
        -0.0979622825,
        -0.1462501260,
        -0.1847425896,
        -0.2073523972,
        -0.2086782295,
        -0.1845719273,
        -0.1326270847,
        -0.0525370892,
        +0.0537187153,
        +0.1818868577,
        +0.3256572849,
        +0.4770745929,
        +0.6271117870,
        +0.7663588857,
        +0.8857664963,
        +0.9773779594,
        +1.0349835419,
        +1.0546365475,
        +1.0349835419,
        +0.9773779594,
        +0.8857664963,
        +0.7663588857,
        +0.6271117870,
        +0.4770745929,
        +0.3256572849,
        +0.1818868577,
        +0.0537187153,
        -0.0525370892,
        -0.1326270847,
        -0.1845719273,
        -0.2086782295,
        -0.2073523972,
        -0.1847425896,
        -0.1462501260,
        -0.0979622825,
        -0.0460635022,
        +0.0037187043,
        +0.0465831968,
        +0.0790009892,
        +0.0989127431,
        +0.1057694293,
        +0.1004237235,
        +0.0848941519,
        +0.0620353691,
        +0.0351552125,
        +0.0076217868,
        -0.0174982391,
        -0.0377352004,
        -0.0514204905,
        -0.0577853377,
        -0.0569528373,
        -0.0498333862,
        -0.0379433607,
        -0.0231737159,
        -0.0075385898,
        +0.0070661879,
        +0.0190682959,
        +0.0273676736,
    ]

    # Symbol mapping based on frequency deviation
    SYMBOLS = np.array([0, -3, -1, 1, 3], dtype=np.int32)
    _SYMBOL_TO_BITS = {3: "01", 1: "00", -1: "10", -3: "11"}

    def __init__(
        self,
        sample_rate=2.4e6,
        center_freq=433.000e6,
        burst_threshold=0.01,
        silence_chunks=5,
        max_burst_duration=0.030,
        enable_plotting=True,
    ):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.decimation = int(self.sample_rate / 48e3)
        self.final_sample_rate = self.sample_rate / self.decimation
        self.sdr = None
        self.rx_stream = None
        self.burst_threshold = burst_threshold
        self.silence_chunks = silence_chunks
        self.max_burst_duration = max_burst_duration
        self.max_samples_in_burst = int(self.sample_rate * self.max_burst_duration)
        self.enable_plotting = enable_plotting
        self.fig = None
        self.ax = None
        self.shift_frequency = (
            self.center_freq - 433.5815e6
        )  # If talkie frequency is 433.575Mhz, center of signal is at 433.582Mhz

    def _setup_plot(self):
        """Initializes the matplotlib figure and axes in interactive mode."""
        if self.enable_plotting:
            print("Plotting enabled. A window will open to display burst analysis.")
            plt.ion()
            self.fig, self.ax = plt.subplots(nrows=3, figsize=(10, 8))
            self.fig.suptitle("DMR Burst Analysis", fontsize=16)
            self.fig.tight_layout(pad=4.0)
            plt.show(block=False)

    def plot_burst_analysis(
        self,
        decimated_signal,
        instant_frequencies,
        sampled_frequencies,
        bins,
        samples_per_symbol,
    ):
        """Updates the plot with data from the most recent burst."""
        if not self.enable_plotting:
            return

        # Clear previous plots
        for axis in self.ax:
            axis.clear()

        # 1. Plot Time-Domain Decimated Signal
        time_axis = np.arange(len(decimated_signal)) / self.final_sample_rate
        self.ax[0].plot(time_axis, np.real(decimated_signal), label="I")
        self.ax[0].plot(time_axis, np.imag(decimated_signal), label="Q", alpha=0.8)
        self.ax[0].set_title("Time Domain (Post-Decimation)")
        self.ax[0].set_xlabel("Time [s]")
        self.ax[0].set_ylabel("Amplitude")
        self.ax[0].legend()
        self.ax[0].grid(True)

        ax_frequency = self.ax[1]
        for i in range(0, len(instant_frequencies), samples_per_symbol * 2):
            f_one_sample = instant_frequencies[i : i + samples_per_symbol * 2]
            time_axis = np.arange(len(f_one_sample))
            ax_frequency.plot(time_axis, f_one_sample)

        ax_frequency.set_title("Instanteneous frequencies")
        ax_frequency.set_ylabel("Instantaneous Frequency [Something])")
        ax_frequency.set_xlabel("Sample Index per Symbol")

        """
        # 2. Plot Eye Diagram
        for i in range(0, len(instant_frequencies), samples_per_symbol * 2):
            segment = instant_frequencies[i : i + samples_per_symbol * 2]
            self.ax[1].plot(np.arange(len(segment)), segment, color='b', alpha=0.3)
        self.ax[1].set_title("Eye Diagram")
        self.ax[1].set_xlabel("Sample Index in Symbol Period")
        self.ax[1].set_ylabel("Instantaneous Frequency")
        self.ax[1].grid(True)
        """

        # 3. Plot Sampled Frequencies and Thresholds
        self.ax[2].plot(sampled_frequencies, "ro", label="Sampled Points")
        for edge in bins:
            self.ax[2].axhline(edge, color="gray", linestyle="--", linewidth=1)
        self.ax[2].set_title("Sampled Symbols and Decision Thresholds")
        self.ax[2].set_xlabel("Symbol Index")
        self.ax[2].set_ylabel("Frequency Value")
        self.ax[2].legend()
        self.ax[2].grid(True)

        # Redraw the canvas
        try:
            self.fig.canvas.flush_events()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Plotting error: {e}")

    def setup_sdr(self):
        """Initializes and configures the SDR."""
        print("Setting up SDR...")
        args = dict(driver="hackrf")
        self.sdr = SoapySDR.Device(args)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        self.sdr.setGain(SOAPY_SDR_RX, 0, "AMP", 0)
        self.sdr.setGain(SOAPY_SDR_RX, 0, "LNA", 16)
        self.sdr.setGain(SOAPY_SDR_RX, 0, "VGA", 16)

        self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rx_stream)
        print("SDR setup complete.")

    """
    def _find_best_phase_offset(self, instant_frequencies, samples_per_symbol):
        L = (len(instant_frequencies) // samples_per_symbol) * samples_per_symbol
        arr = instant_frequencies[:L].reshape(-1, samples_per_symbol).T
        variances = arr.var(axis=1, ddof=0)
        return int(np.argmax(np.sqrt(variances)))
    """

    # pour recentrer exactement sur un burst
    def recenter_array(self, arr):
        target_length = int(
            self.sample_rate * 0.0299
        )  # duration of burst data is 0.0275 sec
        print(target_length)
        return arr[
            (len(arr) - target_length + 1) // 2 : (len(arr) - target_length + 1) // 2
            + target_length
        ]

    # On calcule la variance (déviation standard) pour chaque offset de samples per symbol pour voir lequel on utilise pour sampler les symboles
    def find_best_phase_offset_and_std(self, instant_frequencies, samples_per_symbol):
        L = (len(instant_frequencies) // samples_per_symbol) * samples_per_symbol
        arr = (
            instant_frequencies[:L].reshape(-1, samples_per_symbol).T
        )  # shape (samples_per_symbol, n)
        # en gros on a un tableau à 2 dim, avec sps * de colomnes qui ont une taille du nombre de symbole dans la window (supposé, on découpe en 10)
        variances = arr.var(axis=1, ddof=0)
        std_values = np.sqrt(variances)
        best_offset = int(np.argmax(std_values))
        return best_offset, float(std_values[best_offset])

    def create_frequency_bins(self, instant_frequencies):
        sampled = instant_frequencies

        # compute robust middle using percentiles
        # On va créer nos valeurs limites de bins selon le milieu de nos fréquences (pas forcément 0) pour trier les symboles
        p05, p95 = np.quantile(sampled, [0.05, 0.95])
        middle = (p05 + p95) * 0.5

        # On crée nos bins, donc nos threshhold de choix de symbole
        bins = np.array(
            [
                sampled.min() - 1,
                (sampled.min() + middle) * 0.5,
                middle,
                (sampled.max() + middle) * 0.5,
                sampled.max() + 1,
            ]
        )

        return bins

    def frequencies_to_symbols(self, instant_frequencies, bins):
        # bin indices 1..len(bins)-1; values outside map to 0 or len(bins)
        idx = np.digitize(instant_frequencies, bins, right=True)

        idx_clamped = np.clip(
            idx, 1, len(bins) - 1
        )  # clip pour avoir les values sûr dans les bornes
        symbols = SYMBOLS[idx_clamped]

        # Calculer la zone entre les bins (leur zone de validité quoi, les thresholds)
        left_edges = bins[:-1]
        right_edges = bins[1:]
        middles = (left_edges + right_edges) * 0.5
        half_widths = (right_edges - left_edges) * 0.5

        # index into middles/half_widths using idx_clamped-1
        mid = middles[idx_clamped - 1]
        hw = half_widths[idx_clamped - 1]

        # pour pas avoir de division par 0 si on a un truc qui tombe pile
        with np.errstate(divide="ignore", invalid="ignore"):
            certainty = 1.0 - (np.abs(instant_frequencies - mid) / hw)
        certainty = np.where(hw == 0, 0.0, certainty)
        certainty = np.clip(certainty, 0.0, 1.0)

        out = np.empty(
            instant_frequencies.shape, dtype=[("symbol", "i4"), ("certainty", "f8")]
        )
        out["symbol"] = symbols
        out["certainty"] = certainty
        return out

    # TODO : refractor to handle 8 bits per 8 bits
    def symbols_to_hex(self, symbols):
        # materialize and validate
        arr = np.asarray(list(symbols))
        if arr.size == 0:
            return ""

        try:
            bits_list = [_SYMBOL_TO_BITS[int(s)] for s in arr]
        except KeyError as e:
            raise ValueError(f"invalid symbol: {e}") from None

        bitstream = "".join(bits_list)  # e.g. "010011..." length = 2 * N

        # pad to full bytes (8 bits) by adding '0' bits at the end (LSB side)
        rem = len(bitstream) % 8
        if rem:
            pad = 8 - rem
            bitstream += "0" * pad

        # convert each 8-bit chunk to a byte
        nbytes = len(bitstream) // 8
        bytes_arr = [int(bitstream[i * 8 : (i + 1) * 8], 2) for i in range(nbytes)]

        # format as hex string
        return "".join(f"{b:02x}" for b in bytes_arr)

    def symbols_to_hex2(self, symbols):
        # materialize and validate
        arr = np.asarray(list(symbols))
        if arr.size == 0:
            return ""

        try:
            bits_list = [_SYMBOL_TO_BITS[int(s)] for s in arr]
        except KeyError as e:
            raise ValueError(f"invalid symbol: {e}") from None

        bitstream = "".join(bits_list)  # e.g. "010011..." length = 2 * N

        # pad to full nibbles (4 bits) by adding '0' bits at the end (LSB side)
        rem = len(bitstream) % 4
        if rem:
            pad = 4 - rem
            bitstream += "0" * pad

        # convert each 4-bit chunk to a nibble (0-15)
        nnibbles = len(bitstream) // 4
        nibble_vals = [int(bitstream[i * 4 : (i + 1) * 4], 2) for i in range(nnibbles)]

        # format as hex string (one hex digit per nibble)
        return "".join(f"{n:x}" for n in nibble_vals)

    def detect_sync_pattern(self, hex_string):
        # Extract the substring from index 27 to 39 (108 bit par demis paquets
        potential_sync = hex_string[27:40]
        print(potential_sync)
        print(len(potential_sync))

        # Check for each sync pattern
        for pattern_name, pattern_value in SYNC_PATTERNS.items():
            print(len(pattern_value))
            if pattern_value in potential_sync:
                return pattern_name

        return None

    """
    def _frequencies_to_symbols(self, instant_frequencies):
        p05, p95 = np.quantile(instant_frequencies, [0.05, 0.95])
        middle = (p05 + p95) * 0.5
        bins = np.array([
            instant_frequencies.min() - 1, (instant_frequencies.min() + middle) * 0.5,
            middle, (instant_frequencies.max() + middle) * 0.5,
            instant_frequencies.max() + 1,
        ])
        idx = np.digitize(instant_frequencies, bins, right=True)
        idx_clamped = np.clip(idx, 1, len(bins) - 1)
        symbols = self.SYMBOLS[idx_clamped]
        return symbols, bins

    def _symbols_to_hex(self, symbols):
        if symbols.size == 0: return ""
        try:
            bits_list = [self._SYMBOL_TO_BITS[int(s)] for s in symbols]
        except KeyError as e:
            print(f"Warning: Invalid symbol found: {e}")
            return "Invalid Symbol Detected"
        bitstream = "".join(bits_list)
        rem = len(bitstream) % 8
        if rem: bitstream += "0" * (8 - rem)
        nbytes = len(bitstream) // 8
        bytes_arr = [int(bitstream[i * 8:(i + 1) * 8], 2) for i in range(nbytes)]
        return "".join(f"{b:02x}" for b in bytes_arr)

    """

    """
    def demodulate_burst(self, burst_data):
        print(f"Processing burst of {len(burst_data)} samples...")
        
        low_pass_sos = signal.butter(N=5, Wn=9.6e3, fs=self.sample_rate, output="sos", btype="low")
        filtered_signal = signal.sosfilt(sos=low_pass_sos, x=burst_data)
        decimated_signal = signal.decimate(filtered_signal, q=self.decimation, ftype="fir")
        
        instant_phases = np.unwrap(np.angle(decimated_signal))
        instant_frequencies = np.diff(instant_phases)
        instant_frequencies = signal.lfilter(self.RRCOS_FILTER, 1, instant_frequencies)

        samples_per_symbol = int(self.final_sample_rate / 4800)
        offset = self._find_best_phase_offset(instant_frequencies, samples_per_symbol)
        sampled_frequencies = instant_frequencies[offset::samples_per_symbol]

        if len(sampled_frequencies) < 2:
             print("Burst too short after processing, skipping.")
             return None

        symbols, bins = self._frequencies_to_symbols(sampled_frequencies)
        
        # Call the plotting function with all the calculated data
        self.plot_burst_analysis(decimated_signal, instant_frequencies, sampled_frequencies, bins, samples_per_symbol)
        
        hex_output = self._symbols_to_hex(symbols)
        return hex_output
    """

    def demodulate_burst(self, burst_data):
        # =============== #
        # Low Pass Filter #
        # =============== #

        rx_signal = burst_data

        # We need to apply a low pass filter in order to have a 48kHz sample rate
        """
        low_pass = signal.firwin(
            nb_taps,
            4.8e3,
            fs=sample_rate,
        )
        """
        low_pass = signal.butter(
            N=5, Wn=9.6e3, fs=self.sample_rate, output="sos", btype="low"
        )
        """
        low_pass = signal.cheby1(
            N=5, rp=2, Wn=9.6e3, btype="lowpass", output="sos", fs=sample_rate
        )
        """

        rx_signal = signal.sosfilt(sos=low_pass, x=rx_signal)
        # rx_signal = np.convolve(rx_signal, low_pass, "same")
        # We decimate by sample_rate/48khz, hence sample rate should be a multiple of 48k
        rx_signal = signal.decimate(rx_signal, q=self.decimation, ftype="fir")
        sample_rate = int(self.sample_rate / self.decimation)
        print("SAMPLE RATE AFTER DECIM : " + str(sample_rate))
        print(sample_rate)

        # ================================ #
        # Hamming window (because why not) #
        # ================================ #
        # rx_signal = rx_signal * np.hanning(len(rx_signal))
        # actually it does weird stuff

        # =============================== #
        # Instant frequencies calculation #
        # =============================== #

        # The instant frequency is the derivative of the phase

        samples_per_symbol = sample_rate // 4800
        instant_phases = np.unwrap(np.angle(rx_signal), axis=0)
        instant_frequencies = np.diff(instant_phases)
        # instant_frequencies = np.diff(instant_phases) * (2 * np.pi * (1 / sample_rate))
        # si je transforme en hz, j'ai des valeurs très petites d'IF. Ok chaque sample est très proche mais j'ai "que" 10 sps...
        # en soit ça change rien mais y'a ptetre un pb, les valeurs me paraissent bizarres

        time_axis = np.arange(len(instant_frequencies) // samples_per_symbol)

        # ========================= #
        # Raised Root Cosine Filter #
        # ========================= #

        # Utile pour éviter l'inter symbol interference (et obligatoire car "demi" filtre, appliqué en Rx et Tx)
        # instant_frequencies = np.convolve(rrc_filter, instant_frequencies, "same")
        instant_frequencies = signal.lfilter(self.RRCOS_FILTER, 1, instant_frequencies)

        # ======================================================= #
        #  Standard deviation to chosse sample per symbol to use) #
        # ======================================================= #

        offset, std = self.find_best_phase_offset_and_std(
            instant_frequencies, samples_per_symbol
        )

        sampled_frequencies = instant_frequencies[offset::samples_per_symbol]
        # Add vertical lines for each threshold
        bins = self.create_frequency_bins(sampled_frequencies)

        out = self.frequencies_to_symbols(sampled_frequencies, bins)
        symbols = out["symbol"]
        hex_output = self.symbols_to_hex2(symbols)

        self.plot_burst_analysis(
            rx_signal,
            instant_frequencies,
            sampled_frequencies,
            bins,
            samples_per_symbol,
        )

        return hex_output


    def get_good_burst_start(self, rx_signal):
        pass

    def process_stream(self):
        self.setup_sdr()
        self._setup_plot()

        buff_len = 128
        chunks_per_burst = ceil((self.sample_rate * 0.0275) / buff_len)
        rx_buff = np.array([0] * buff_len, np.complex64)
        in_burst, burst_buffer, silence_counter, samples_in_burst = False, [], 0, 0

        print("\n--- Starting Stream Demodulation ---")
        print(f"Listening for bursts with power > {self.burst_threshold}")

        try:
            while True:
                sr = self.sdr.readStream(
                    self.rx_stream, [rx_buff], buff_len, timeoutUs=5000000
                )
                if sr.ret < 0:
                    print(f"Error reading stream: {sr.ret}")
                    continue

                t = np.arange(len(rx_buff[: sr.ret])) / self.sample_rate
                mixing_signal = np.exp(1j * 2 * np.pi * self.shift_frequency * t)
                rx_signal = rx_buff[: sr.ret] * mixing_signal
                power = np.mean(np.abs(rx_signal) ** 2)
                process_this_burst = False

                if in_burst:
                    burst_buffer.append(rx_signal.copy())
                    samples_in_burst += sr.ret
                    if power < self.burst_threshold:
                        silence_counter += 1
                        if silence_counter > self.silence_chunks:
                            print("Burst ended (silence).")
                            process_this_burst = True
                            burst_buffer = burst_buffer[: -self.silence_chunks]
                    else:
                        silence_counter = 0
                    if (
                        not process_this_burst
                        and samples_in_burst >= self.max_samples_in_burst
                    ):
                        print(
                            f"Burst ended (hit {self.max_burst_duration * 1000}ms time limit)."
                        )
                        process_this_burst = True
                elif power > self.burst_threshold:
                    print(f"--- Burst detected! (Power: {power:.4f}) ---")
                    in_burst, samples_in_burst, silence_counter = True, sr.ret, 0
                    burst_buffer = [rx_signal.copy()]

                    # on recentre le burst et on garde le nombre de sample exact pour le nombre de symboles

                if process_this_burst:
                    full_burst = np.concatenate(burst_buffer)
                    #print(full_burst)
                    #full_burst = self.recenter_array(full_burst)
                    hex_data = self.demodulate_burst(full_burst)
                    if hex_data:
                        print(f"--> HEX Output: {hex_data}\n")
                        sync = self.detect_sync_pattern(hex_data)
                        print(sync)
                    in_burst = False
                    print(f"Listening for bursts with power > {self.burst_threshold}")

        except KeyboardInterrupt:
            print("\nUser interrupted. Shutting down.")
        finally:
            print("Deactivating stream...")
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)


if __name__ == "__main__":
    demodulator = DMRStreamDemodulator(
        sample_rate=2.4e6,
        center_freq=433.000e6,
        burst_threshold=0.2,
        silence_chunks=5,
        max_burst_duration=0.028,
        enable_plotting=True,
    )
    demodulator.process_stream()
