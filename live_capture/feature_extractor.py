import time
import numpy as np
import pandas as pd


class FlowStats:
    """
    Collect statistics for one bidirectional network flow.

    The packet_capture.py file is responsible for identifying
    the flow and determining packet direction.

    This class extracts the 52 features used by SecureNet.
    """

    def __init__(self, destination_port):

        self.destination_port = int(destination_port)

        # --------------------------------------------------
        # Flow timing
        # --------------------------------------------------

        self.start_time = None
        self.last_time = None

        # --------------------------------------------------
        # Packet storage
        # --------------------------------------------------

        self.forward_packets = []
        self.backward_packets = []

        self.forward_timestamps = []
        self.backward_timestamps = []

        # --------------------------------------------------
        # Header lengths
        # --------------------------------------------------

        self.forward_header_lengths = []
        self.backward_header_lengths = []

        # --------------------------------------------------
        # TCP
        # --------------------------------------------------

        self.fin_count = 0
        self.psh_count = 0
        self.ack_count = 0

        self.init_win_forward = 0
        self.init_win_backward = 0

        # --------------------------------------------------
        # Forward data packets
        # --------------------------------------------------

        self.forward_data_packets = 0
        self.forward_segment_sizes = []

        # --------------------------------------------------
        # Active / Idle periods
        # --------------------------------------------------

        self.active_times = []
        self.idle_times = []

    # ======================================================
    # Add packet
    # ======================================================

    def add_packet(
        self,
        packet_length,
        timestamp,
        direction="forward",
        header_length=0,
        tcp_window=0,
        fin=False,
        psh=False,
        ack=False
    ):

        packet_length = int(packet_length)
        timestamp = float(timestamp)
        header_length = int(header_length)
        tcp_window = int(tcp_window)

        # --------------------------------------------------
        # First packet
        # --------------------------------------------------

        if self.start_time is None:
            self.start_time = timestamp

        self.last_time = timestamp

        # --------------------------------------------------
        # Forward packet
        # --------------------------------------------------

        if direction == "forward":

            self.forward_packets.append(packet_length)
            self.forward_timestamps.append(timestamp)
            self.forward_header_lengths.append(header_length)

            # Initial TCP window
            if self.init_win_forward == 0 and tcp_window > 0:
                self.init_win_forward = tcp_window

            # Data packet
            payload_length = packet_length - header_length

            if payload_length > 0:
                self.forward_data_packets += 1
                self.forward_segment_sizes.append(payload_length)

        # --------------------------------------------------
        # Backward packet
        # --------------------------------------------------

        else:

            self.backward_packets.append(packet_length)
            self.backward_timestamps.append(timestamp)
            self.backward_header_lengths.append(header_length)

            # Initial TCP window
            if self.init_win_backward == 0 and tcp_window > 0:
                self.init_win_backward = tcp_window

        # --------------------------------------------------
        # TCP flags
        # --------------------------------------------------

        if fin:
            self.fin_count += 1

        if psh:
            self.psh_count += 1

        if ack:
            self.ack_count += 1

    # ======================================================
    # Utility functions
    # ======================================================

    @staticmethod
    def mean(values):

        if len(values) == 0:
            return 0.0

        return float(np.mean(values))

    @staticmethod
    def std(values):

        if len(values) <= 1:
            return 0.0

        return float(np.std(values))

    @staticmethod
    def minimum(values):

        if len(values) == 0:
            return 0.0

        return float(np.min(values))

    @staticmethod
    def maximum(values):

        if len(values) == 0:
            return 0.0

        return float(np.max(values))

    # ======================================================
    # Calculate active / idle periods
    # ======================================================

    @staticmethod
    def calculate_active_idle(timestamps):

        if len(timestamps) < 2:
            return [], []

        timestamps = sorted(timestamps)

        active = []
        idle = []

        current_active = 0.0

        for i in range(1, len(timestamps)):

            gap = (
                timestamps[i] -
                timestamps[i - 1]
            ) * 1_000_000

            # CICIDS/CICFlowMeter-style threshold:
            # 1 second = 1,000,000 microseconds

            if gap > 1_000_000:

                if current_active > 0:
                    active.append(current_active)

                idle.append(gap)

                current_active = 0.0

            else:

                current_active += gap

        if current_active > 0:
            active.append(current_active)

        return active, idle

    # ======================================================
    # Generate 52 ML features
    # ======================================================

    def to_features(self):

        # --------------------------------------------------
        # All packets
        # --------------------------------------------------

        all_packets = (
            self.forward_packets +
            self.backward_packets
        )

        total_packets = len(all_packets)

        total_fwd = len(self.forward_packets)
        total_bwd = len(self.backward_packets)

        total_fwd_bytes = sum(self.forward_packets)
        total_bwd_bytes = sum(self.backward_packets)

        total_bytes = (
            total_fwd_bytes +
            total_bwd_bytes
        )

        # --------------------------------------------------
        # Timestamps
        # --------------------------------------------------

        all_timestamps = sorted(
            self.forward_timestamps +
            self.backward_timestamps
        )

        # --------------------------------------------------
        # Flow duration
        # --------------------------------------------------

        if len(all_timestamps) >= 2:

            duration = (
                all_timestamps[-1] -
                all_timestamps[0]
            ) * 1_000_000

        else:

            duration = 0.0

        # --------------------------------------------------
        # Flow IAT
        # --------------------------------------------------

        if len(all_timestamps) >= 2:

            flow_iats = (
                np.diff(all_timestamps) *
                1_000_000
            )

        else:

            flow_iats = np.array([])

        # --------------------------------------------------
        # Forward IAT
        # --------------------------------------------------

        if len(self.forward_timestamps) >= 2:

            fwd_iats = (
                np.diff(
                    self.forward_timestamps
                ) *
                1_000_000
            )

        else:

            fwd_iats = np.array([])

        # --------------------------------------------------
        # Backward IAT
        # --------------------------------------------------

        if len(self.backward_timestamps) >= 2:

            bwd_iats = (
                np.diff(
                    self.backward_timestamps
                ) *
                1_000_000
            )

        else:

            bwd_iats = np.array([])

        # --------------------------------------------------
        # Active / Idle
        # --------------------------------------------------

        self.active_times, self.idle_times = (
            self.calculate_active_idle(
                all_timestamps
            )
        )

        # --------------------------------------------------
        # Rates
        # --------------------------------------------------

        if duration > 0:

            duration_seconds = (
                duration / 1_000_000
            )

            flow_bytes_per_sec = (
                total_bytes /
                duration_seconds
            )

            flow_packets_per_sec = (
                total_packets /
                duration_seconds
            )

            fwd_packets_per_sec = (
                total_fwd /
                duration_seconds
            )

            bwd_packets_per_sec = (
                total_bwd /
                duration_seconds
            )

        else:

            flow_bytes_per_sec = 0.0
            flow_packets_per_sec = 0.0
            fwd_packets_per_sec = 0.0
            bwd_packets_per_sec = 0.0

        # --------------------------------------------------
        # 52 FEATURES
        # --------------------------------------------------

        features = {

            "Destination Port":
                self.destination_port,

            "Flow Duration":
                duration,

            "Total Fwd Packets":
                total_fwd,

            "Total Length of Fwd Packets":
                total_fwd_bytes,

            "Fwd Packet Length Max":
                self.maximum(
                    self.forward_packets
                ),

            "Fwd Packet Length Min":
                self.minimum(
                    self.forward_packets
                ),

            "Fwd Packet Length Mean":
                self.mean(
                    self.forward_packets
                ),

            "Fwd Packet Length Std":
                self.std(
                    self.forward_packets
                ),

            "Bwd Packet Length Max":
                self.maximum(
                    self.backward_packets
                ),

            "Bwd Packet Length Min":
                self.minimum(
                    self.backward_packets
                ),

            "Bwd Packet Length Mean":
                self.mean(
                    self.backward_packets
                ),

            "Bwd Packet Length Std":
                self.std(
                    self.backward_packets
                ),

            "Flow Bytes/s":
                flow_bytes_per_sec,

            "Flow Packets/s":
                flow_packets_per_sec,

            "Flow IAT Mean":
                self.mean(flow_iats),

            "Flow IAT Std":
                self.std(flow_iats),

            "Flow IAT Max":
                self.maximum(flow_iats),

            "Flow IAT Min":
                self.minimum(flow_iats),

            "Fwd IAT Total":
                float(np.sum(fwd_iats))
                if len(fwd_iats) > 0
                else 0.0,

            "Fwd IAT Mean":
                self.mean(fwd_iats),

            "Fwd IAT Std":
                self.std(fwd_iats),

            "Fwd IAT Max":
                self.maximum(fwd_iats),

            "Fwd IAT Min":
                self.minimum(fwd_iats),

            "Bwd IAT Total":
                float(np.sum(bwd_iats))
                if len(bwd_iats) > 0
                else 0.0,

            "Bwd IAT Mean":
                self.mean(bwd_iats),

            "Bwd IAT Std":
                self.std(bwd_iats),

            "Bwd IAT Max":
                self.maximum(bwd_iats),

            "Bwd IAT Min":
                self.minimum(bwd_iats),

            "Fwd Header Length":
                sum(
                    self.forward_header_lengths
                ),

            "Bwd Header Length":
                sum(
                    self.backward_header_lengths
                ),

            "Fwd Packets/s":
                fwd_packets_per_sec,

            "Bwd Packets/s":
                bwd_packets_per_sec,

            "Min Packet Length":
                self.minimum(
                    all_packets
                ),

            "Max Packet Length":
                self.maximum(
                    all_packets
                ),

            "Packet Length Mean":
                self.mean(
                    all_packets
                ),

            "Packet Length Std":
                self.std(
                    all_packets
                ),

            "Packet Length Variance":
                float(np.var(all_packets))
                if len(all_packets) > 0
                else 0.0,

            "FIN Flag Count":
                self.fin_count,

            "PSH Flag Count":
                self.psh_count,

            "ACK Flag Count":
                self.ack_count,

            "Average Packet Size":
                (
                    total_bytes /
                    total_packets
                )
                if total_packets > 0
                else 0.0,

            "Subflow Fwd Bytes":
                total_fwd_bytes,

            "Init_Win_bytes_forward":
                self.init_win_forward,

            "Init_Win_bytes_backward":
                self.init_win_backward,

            "act_data_pkt_fwd":
                self.forward_data_packets,

            "min_seg_size_forward":
                self.minimum(
                    self.forward_segment_sizes
                ),

            "Active Mean":
                self.mean(
                    self.active_times
                ),

            "Active Max":
                self.maximum(
                    self.active_times
                ),

            "Active Min":
                self.minimum(
                    self.active_times
                ),

            "Idle Mean":
                self.mean(
                    self.idle_times
                ),

            "Idle Max":
                self.maximum(
                    self.idle_times
                ),

            "Idle Min":
                self.minimum(
                    self.idle_times
                ),
        }

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if len(features) != 52:

            raise ValueError(
                f"Expected 52 features, "
                f"but generated {len(features)}"
            )

        return pd.DataFrame([features])