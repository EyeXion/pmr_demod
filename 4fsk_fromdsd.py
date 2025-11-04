#!/usr/bin/env python3
"""
DMR (Digital Mobile Radio) Decoder - Terminal Version
Outputs decoded packets in hexadecimal format

Original code from DSD software
Copyright (C) 2010 DSD Author
GPG Key ID: 0x3F1D7FD0
"""

from datetime import datetime
from typing import List
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


class DMRDecode:
    """Main DMR decoder class - terminal output only"""

    # Constants
    SAMPLESPERSYMBOL = 10
    SYMBOLCENTRE = 4
    MAXSTARTVALUE = 15000
    MINSTARTVALUE = -15000
    SYMBOLSAHEAD = 144
    SAMPLESAHEADSIZE = (SYMBOLSAHEAD * SAMPLESPERSYMBOL) + SAMPLESPERSYMBOL
    JITTERFRAMEADJUST = 1
    JITTERCOUNTERSIZE = JITTERFRAMEADJUST * 144
    MAXMINBUFSIZE = 5

    # DMR Sync patterns
    DMR_DATA_SYNC_BS = [
        3,
        1,
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        3,
        3,
        1,
        1,
        3,
        1,
        1,
        3,
        1,
        3,
        3,
        1,
        1,
        3,
        1,
    ]
    DMR_VOICE_SYNC_BS = [
        1,
        3,
        1,
        1,
        1,
        1,
        3,
        3,
        3,
        1,
        1,
        3,
        3,
        1,
        3,
        3,
        1,
        3,
        1,
        1,
        3,
        3,
        1,
        3,
    ]
    DMR_DATA_SYNC_MS = [
        3,
        1,
        1,
        1,
        3,
        1,
        1,
        3,
        3,
        3,
        1,
        3,
        1,
        3,
        3,
        3,
        3,
        1,
        1,
        3,
        1,
        1,
        1,
        3,
    ]
    DMR_VOICE_SYNC_MS = [
        1,
        3,
        3,
        3,
        1,
        3,
        3,
        1,
        1,
        1,
        3,
        1,
        3,
        1,
        1,
        1,
        1,
        3,
        3,
        1,
        3,
        3,
        3,
        1,
    ]
    DMR_RC_SYNC = [
        1,
        3,
        1,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        1,
        3,
        3,
        1,
        3,
        3,
        3,
        1,
        1,
        3,
        1,
        3,
    ]
    DMR_DATA_SYNC_DIRECT1 = [
        3,
        3,
        1,
        3,
        3,
        3,
        3,
        1,
        3,
        1,
        1,
        1,
        3,
        1,
        3,
        1,
        3,
        3,
        3,
        1,
        1,
        1,
        1,
        1,
    ]
    DMR_VOICE_SYNC_DIRECT1 = [
        1,
        1,
        3,
        1,
        1,
        1,
        1,
        3,
        1,
        3,
        3,
        3,
        1,
        3,
        1,
        3,
        1,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
    ]
    DMR_DATA_SYNC_DIRECT2 = [
        3,
        1,
        1,
        3,
        1,
        1,
        1,
        1,
        1,
        3,
        3,
        3,
        1,
        1,
        3,
        3,
        3,
        3,
        1,
        3,
        3,
        3,
        1,
        1,
    ]
    DMR_VOICE_SYNC_DIRECT2 = [
        1,
        3,
        3,
        1,
        3,
        3,
        3,
        3,
        3,
        1,
        1,
        1,
        3,
        3,
        1,
        1,
        1,
        1,
        3,
        1,
        1,
        1,
        3,
        3,
    ]
    DMR_DATA_REST_SYNC_BS = [
        3,
        1,
        3,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
        1,
        1,
        3,
        1,
        1,
        3,
        1,
        1,
        1,
        3,
        3,
        1,
        3,
        1,
    ]

    def __init__(self):
        self.program_version = "DMR Decoder Python - Terminal v1.0"

        # State variables
        self.max_val = self.MAXSTARTVALUE
        self.min_val = self.MINSTARTVALUE
        self.centre = 0
        self.lmid = 0
        self.umid = 0
        self.lastsynctype = -1
        self.symbolcnt = 0
        self.carrier = False
        self.inverted = True
        self.firstframe = False
        self.frameSync = False
        self.jitter = -1
        self.lastSample = 0
        self.mode = -1

        # Buffers
        self.dibit_circular_buffer = [0] * 144
        self.dibit_circular_buffer_counter = 0
        self.dibit_frame = [0] * 144
        self.symbol_buffer = [0] * 144
        self.symbol_buffer_counter = 0
        self.samples_ahead_buffer = [0] * self.SAMPLESAHEADSIZE
        self.samples_ahead_counter = 0
        self.jitter_buffer = [0] * self.JITTERCOUNTERSIZE
        self.jitter_counter = 0
        self.sync_high_low_buf = [0] * 24
        self.max_buffer = [0] * self.MAXMINBUFSIZE
        self.min_buffer = [0] * self.MAXMINBUFSIZE
        self.maxmin_buffer_counter = 0

        # Statistics
        self.frame_count = 0
        self.bad_frame_count = 0
        self.embedded_frame_count = 0

        # User-provided sample function
        self.get_sample_func = None

    def set_sample_source(self, func):
        """Set the function to call for getting samples"""
        self.get_sample_func = func

    def get_sample_internal(self) -> int:
        """Get the next audio sample from user function"""
        if self.get_sample_func is None:
            raise Exception(
                "No sample source configured! Call set_sample_source() first."
            )

        sample = self.get_sample_func()

        # Add to samples ahead buffer
        self.samples_ahead_buffer[self.samples_ahead_counter] = sample
        self.samples_ahead_counter += 1
        if self.samples_ahead_counter == self.SAMPLESAHEADSIZE:
            self.samples_ahead_counter = 0

        # Return oldest sample from buffer
        return self.samples_ahead_buffer[self.samples_ahead_counter]

    def get_symbol(self, have_sync: bool) -> int:
        """Convert SAMPLESPERSYMBOL samples into a single symbol"""
        symbol_sum = 0
        count = 0

        for i in range(self.SAMPLESPERSYMBOL):
            # Handle jitter adjustment
            if i == 0 and self.jitter > 0:
                if self.jitter > 0 and self.jitter <= self.SYMBOLCENTRE:
                    i -= 1
                elif (
                    self.jitter > self.SYMBOLCENTRE
                    and self.jitter < self.SAMPLESPERSYMBOL
                ):
                    i += 1
                self.jitter = -1

            sample = self.get_sample_internal()

            # Jitter detection on zero crossings
            if sample > self.centre:
                if self.lastSample < self.centre:
                    if not self.frameSync:
                        self.jitter = i
                    else:
                        self.process_jitter(i)
            else:
                if self.lastSample > self.centre:
                    if not self.frameSync:
                        self.jitter = i
                    else:
                        self.process_jitter(i)

            # Sample at symbol centre
            if self.SYMBOLCENTRE <= i <= self.SYMBOLCENTRE + 1:
                symbol_sum += sample
                count += 1

            self.lastSample = sample

        self.symbolcnt += 1
        return symbol_sum // count if count > 0 else 0

    def process_jitter(self, jit: int):
        """Add jitter value to buffer and calculate mode"""
        self.jitter_buffer[self.jitter_counter] = jit
        self.jitter_counter += 1

        if self.jitter_counter == self.JITTERCOUNTERSIZE:
            self.jitter_counter = 0
            self.jitter = self.calc_jitter_mode()

    def calc_jitter_mode(self) -> int:
        """Calculate the mode (most common value) of jitter buffer"""
        counts = [0] * self.SAMPLESPERSYMBOL

        for val in self.jitter_buffer:
            if 0 <= val < self.SAMPLESPERSYMBOL:
                counts[val] += 1

        return counts.index(max(counts))

    def symbol_to_dibit(self, symbol: int) -> int:
        """Convert a symbol to a 2-bit dibit value"""
        if self.frameSync:
            if not self.inverted:
                if symbol > self.centre:
                    return 1 if symbol > self.umid else 0
                else:
                    return 3 if symbol < self.lmid else 2
            else:
                if symbol > self.centre:
                    return 3 if symbol > self.umid else 2
                else:
                    return 1 if symbol < self.lmid else 0
        else:
            if not self.inverted:
                return 1 if symbol > 0 else 3
            else:
                return 3 if symbol > 0 else 1

    def add_to_dibit_buffer(self, dibit: int):
        """Add a dibit to the circular buffer"""
        self.dibit_circular_buffer[self.dibit_circular_buffer_counter] = dibit
        self.dibit_circular_buffer_counter += 1
        if self.dibit_circular_buffer_counter == 144:
            self.dibit_circular_buffer_counter = 0

    def add_to_symbol_buffer(self, symbol: int):
        """Add a symbol to the circular buffer"""
        self.symbol_buffer[self.symbol_buffer_counter] = symbol
        self.symbol_buffer_counter += 1
        if self.symbol_buffer_counter == 144:
            self.symbol_buffer_counter = 0

    def get_sync_symbols(self) -> List[int]:
        """Extract the 24 sync symbols from the symbol buffer"""
        syms = []
        circ_pos = self.symbol_buffer_counter + 66
        if circ_pos >= 144:
            circ_pos -= 144

        for i in range(24):
            syms.append(self.symbol_buffer[circ_pos])
            circ_pos += 1
            if circ_pos == 144:
                circ_pos = 0

        return syms

    def sync_compare(self, sync: bool) -> int:
        """
        Compare sync patterns and return type:
        0=unknown, 1=BS voice, 2=BS data, 3=MS voice, 4=MS data,
        5=RC sync, 6=Direct voice 1, 7=Direct data 1,
        8=Direct voice 2, 9=Direct data 2, 10=Rest data
        """
        diff = 5 if sync else 0

        circ_pos = self.dibit_circular_buffer_counter + 66
        if circ_pos >= 144:
            circ_pos -= 144

        scores = {
            "voice_bs": 0,
            "data_bs": 0,
            "voice_ms": 0,
            "data_ms": 0,
            "rc": 0,
            "voice_d1": 0,
            "data_d1": 0,
            "voice_d2": 0,
            "data_d2": 0,
            "rest": 0,
        }

        for i in range(24):
            dib = self.dibit_circular_buffer[circ_pos]

            if dib == self.DMR_VOICE_SYNC_BS[i]:
                scores["voice_bs"] += 1
            if dib == self.DMR_DATA_SYNC_BS[i]:
                scores["data_bs"] += 1
            if dib == self.DMR_VOICE_SYNC_MS[i]:
                scores["voice_ms"] += 1
            if dib == self.DMR_DATA_SYNC_MS[i]:
                scores["data_ms"] += 1
            if dib == self.DMR_RC_SYNC[i]:
                scores["rc"] += 1
            if dib == self.DMR_VOICE_SYNC_DIRECT1[i]:
                scores["voice_d1"] += 1
            if dib == self.DMR_DATA_SYNC_DIRECT1[i]:
                scores["data_d1"] += 1
            if dib == self.DMR_VOICE_SYNC_DIRECT2[i]:
                scores["voice_d2"] += 1
            if dib == self.DMR_DATA_SYNC_DIRECT2[i]:
                scores["data_d2"] += 1
            if dib == self.DMR_DATA_REST_SYNC_BS[i]:
                scores["rest"] += 1

            circ_pos += 1
            if circ_pos == 144:
                circ_pos = 0

        # Check scores against thresholds
        if (24 - scores["voice_bs"]) <= diff:
            return 1
        if (24 - scores["data_bs"]) <= diff:
            return 2
        if (24 - scores["voice_ms"]) <= diff:
            return 3
        if (24 - scores["data_ms"]) <= diff:
            return 4
        if (24 - scores["rc"]) <= diff:
            return 5
        if (24 - scores["voice_d1"]) <= diff:
            return 6
        if (24 - scores["data_d1"]) <= diff:
            return 7
        if (24 - scores["voice_d2"]) <= diff:
            return 8
        if (24 - scores["data_d2"]) <= diff:
            return 9
        if (24 - scores["rest"]) <= diff:
            return 10

        return 0

    def frame_calcs(self, lmin: int, lmax: int):
        """Calculate frame parameters from min/max values"""
        self.max_val = lmax
        self.min_val = lmin
        self.centre = (self.max_val + self.min_val) // 2
        self.umid = int((self.max_val - self.centre) * 0.625) + self.centre
        self.lmid = int((self.min_val - self.centre) * 0.625) + self.centre

    def add_to_min_max_buffer(self, lmin: int, lmax: int):
        """Add min/max values to circular buffer"""
        self.max_buffer[self.maxmin_buffer_counter] = lmax
        self.min_buffer[self.maxmin_buffer_counter] = lmin
        self.maxmin_buffer_counter += 1

        if self.maxmin_buffer_counter == self.MAXMINBUFSIZE:
            self.maxmin_buffer_counter = 0
            self.calc_average_min_max()

    def calc_average_min_max(self):
        """Calculate average min/max from buffer"""
        avg_max = sum(self.max_buffer) // self.MAXMINBUFSIZE
        avg_min = sum(self.min_buffer) // self.MAXMINBUFSIZE
        self.jitter_counter = 0
        self.frame_calcs(avg_min, avg_max)

    def get_frame_sync(self) -> int:
        """
        Look for frame sync pattern and return sync type
        Returns -1 if no sync found
        """
        t = 0
        synctest_pos = 0
        self.symbolcnt = 0

        while True:
            t += 1

            # Get symbol and add to buffer
            symbol = self.get_symbol(self.frameSync)
            self.add_to_symbol_buffer(symbol)

            # Convert to dibit and add to buffer
            dibit = self.symbol_to_dibit(symbol)
            self.add_to_dibit_buffer(dibit)

            # Check for sync after 144 dibits
            if t >= 144:
                if not self.frameSync or (self.frameSync and self.symbolcnt % 144 == 0):
                    # Get sync symbols and find min/max
                    self.sync_high_low_buf = self.get_sync_symbols()
                    lmin = min(self.sync_high_low_buf)
                    lmax = max(self.sync_high_low_buf)

                # Check for sync pattern
                if not self.frameSync or (self.frameSync and self.symbolcnt % 144 == 0):
                    sync_type = self.sync_compare(self.frameSync)

                    # Embedded signaling frame
                    if (
                        self.frameSync
                        and sync_type == 0
                        and not self.firstframe
                        and self.embedded_frame_count < 7
                    ):
                        self.embedded_frame_count += 1
                        self.lastsynctype = 13
                        return 13

                    # Handle different sync types
                    if sync_type > 0:
                        self.carrier = True

                        if not self.frameSync:
                            self.frame_calcs(lmin, lmax)
                            self.frameSync = True
                        else:
                            self.add_to_min_max_buffer(lmin, lmax)

                        self.firstframe = self.lastsynctype == -1

                        # Set mode based on sync type
                        if sync_type in [1, 2, 10]:  # BS
                            self.mode = 0
                            self.embedded_frame_count = 0
                        elif sync_type in [3, 4, 5]:  # MS
                            self.mode = 1
                            self.embedded_frame_count = 0
                        elif sync_type in [6, 7, 8, 9]:  # Direct
                            self.mode = 2
                            self.embedded_frame_count = 0

                        # Map sync types to return values
                        sync_map = {
                            1: 12,
                            2: 10,
                            3: 22,
                            4: 20,
                            5: 25,
                            6: 30,
                            7: 31,
                            8: 32,
                            9: 33,
                            10: 40,
                        }
                        self.lastsynctype = sync_map.get(sync_type, 0)
                        return self.lastsynctype

            # Check for lost sync
            if self.carrier and synctest_pos >= (144 * 12):
                self.frameSync = False
                self.no_carrier()
                return -1

            # Reset if search goes too long
            if t > 32000:
                t = 0
                synctest_pos = 0
            else:
                synctest_pos += 1

    def create_dibit_frame(self):
        """Create ordered frame from circular buffer"""
        circ_pos = self.dibit_circular_buffer_counter - 144
        if circ_pos < 0:
            circ_pos = 144 + circ_pos

        for i in range(144):
            self.dibit_frame[i] = self.dibit_circular_buffer[circ_pos]
            circ_pos += 1
            if circ_pos == 144:
                circ_pos = 0

    def dibits_to_hex(self) -> str:
        """Convert dibit frame to hexadecimal string"""
        # Each byte contains 4 dibits (8 bits)
        hex_bytes = []
        for i in range(0, 144, 4):
            # Pack 4 dibits into one byte
            byte_val = (
                (self.dibit_frame[i] << 6)
                | (self.dibit_frame[i + 1] << 4)
                | (self.dibit_frame[i + 2] << 2)
                | (self.dibit_frame[i + 3])
            )
            hex_bytes.append(byte_val)

        # Convert to hex string
        return " ".join(f"{b:02X}" for b in hex_bytes)

    def get_sync_type_name(self, sync_type: int) -> str:
        """Get human-readable sync type name"""
        names = {
            10: "BS_DATA",
            12: "BS_VOICE",
            13: "EMBEDDED",
            20: "MS_DATA",
            22: "MS_VOICE",
            25: "RC_SYNC",
            30: "DIRECT_VOICE_1",
            31: "DIRECT_DATA_1",
            32: "DIRECT_VOICE_2",
            33: "DIRECT_DATA_2",
            40: "REST_DATA",
        }
        return names.get(sync_type, "UNKNOWN")

    def process_frame(self):
        """Process a detected DMR frame"""
        if self.firstframe:
            self.maxmin_buffer_counter = 0
            print(
                f"\n[{self.get_timestamp()}] SYNC ACQUIRED: {self.get_sync_type_name(self.synctype)}"
            )
            return

        self.frame_count += 1

        # Print frame information
        timestamp = self.get_timestamp()
        sync_name = self.get_sync_type_name(self.synctype)
        hex_data = self.dibits_to_hex()

        print(
            f"[{timestamp}] Frame #{self.frame_count:04d} | {sync_name:15s} | {hex_data}"
        )

    def no_carrier(self):
        """Reset state when carrier is lost"""
        if self.carrier:  # Only print if we had a carrier
            print(f"\n[{self.get_timestamp()}] SYNC LOST\n")

        self.jitter = -1
        self.lastsynctype = -1
        self.carrier = False
        self.max_val = self.MAXSTARTVALUE
        self.min_val = self.MINSTARTVALUE
        self.centre = 0
        self.firstframe = False
        self.mode = -1

    def decode(self):
        """Main decode loop - call this continuously"""
        self.no_carrier()
        self.synctype = self.get_frame_sync()

        while self.synctype != -1:
            self.process_frame()
            self.synctype = self.get_frame_sync()
            self.create_dibit_frame()

    def get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# Example usage
def example_usage():
    """
    Example of how to use the decoder with a custom sample source
    """
    import random

    # ============== #
    # SDR parameters #
    # ============== #

    sample_rate = 4.8e6
    # Duration of the recording [s].
    D = 1
    # Number of samples to record.
    N = int(D * sample_rate)
    center_freq = 433.000e6
    # Decimation value after first low pass filter
    decimation = int(sample_rate / 48e3)

    # ============================ #
    # Figure and time window setup #
    # ============================ #

    window_duration = 0.028
    window_offset = 0

    fig, ax = plt.subplots(nrows=3)
    fig.subplots_adjust(left=0.15, right=0.85)
    axwindow_duration = fig.add_axes([0.05, 0.25, 0.01, 0.65])
    window_duration_slider = Slider(
        ax=axwindow_duration,
        label="Window duration [s]",
        valmin=1 / 4800,
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

    # ========= #
    # SDR setup #
    # ========= #

    # enumerate devices
    results = SoapySDR.Device.enumerate()
    for result in results:
        print(result)

    # create device instance
    # args can be user defined or from the enumeration result
    args = dict(driver="hackrf")
    # args = dict(driver="lime")
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

    # =========== #
    # SDR capture #
    # =========== #

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

    # ============ #
    # Save capture #
    # ============ #

    date = datetime.now().strftime("%Y%m%d_%H_%M_%S%m")
    rx_signal_path = (
        "samples/DMRCapture_HackRF_" + str(int(sample_rate)) + "Hz_" + date + ".npy"
    )

    # print("Signal samples: {}".format(len(rx_signal)))
    # print("Save to {}".format(rx_signal_path))
    # np.save(rx_signal_path, rx_signal)

    # Uncomment to load a file...
    # rx_signal = np.load("DMRCapture_HackRF_2400000Hz_20250922_09_32_4709.npy")

    # ====== #
    # Mixing #
    # ====== #
    t = np.arange(len(rx_signal)) / sample_rate
    shift_frequency = (
        center_freq - 433.5815e6
    )  # If talkie frequency is 433.575Mhz, center of signal is at 433.582Mhz
    mixing_signal = np.exp(1j * 2 * np.pi * shift_frequency * t)
    rx_signal = rx_signal * mixing_signal

    rx_signal_full = rx_signal.copy()
    original_sample_rate = sample_rate

    # Example: Create a simple sample generator (replace with your actual source)
    def get_sample():
        """
        This function should return the next audio sample as a signed 16-bit integer
        Replace this with your actual sample source (SDR, file, etc.)
        """
        # Simulated random samples (replace with real data!)
        return random.randint(-15000, 15000)

    # Create decoder instance
    decoder = DMRDecode()

    # Set the sample source
    decoder.set_sample_source(get_sample)

    print(f"=== {decoder.program_version} ===")
    print("Waiting for DMR signal...\n")

    try:
        # Run the decoder (this will loop forever)
        while True:
            decoder.decode()
    except KeyboardInterrupt:
        print("\n\nDecoder stopped by user.")


if __name__ == "__main__":
    example_usage()
