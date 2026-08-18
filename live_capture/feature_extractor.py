import time
import numpy as np
import pandas as pd


class FlowStats:
    """
    Collect statistics for one network flow.

    A flow is identified externally by:
        source IP
        destination IP
        source port
        destination port
        protocol
    """

    def __init__(self, src_port, dst_port):
        self.src_port = src_port
        self.dst_port = dst_port

        self.start_time = time.time()
        self.last_time = self.start_time

        self.forward_packets = []
        self.backward_packets = []

        self.forward_timestamps = []
        self.backward_timestamps = []

        self.forward_lengths = []
        self.backward_lengths = []

        self.fin_count = 0
        self.psh_count = 0
        self.ack_count = 0

        self.init_win_forward = 0
        self.init_win_backward = 0

        self.forward_header_lengths = []
        self.backward_header_lengths = []

        self.active_times = []
        self.idle_times = []

    # --------------------------------------------------
    # Add packet
    # --------------------------------------------------

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

        if direction == "forward":

            self.forward_packets.append(packet_length)
            self.forward_timestamps.append(timestamp)
            self.forward_lengths.append(packet_length)
            self.forward_header_lengths.append(header_length)

            if self.init_win_forward == 0:
                self.init_win_forward = tcp_window

        else:

            self.backward_packets.append(packet_length)
            self.backward_timestamps.append(timestamp)
            self.backward_lengths.append(packet_length)
            self.backward_header_lengths.append(header_length)

            if self.init_win_backward == 0:
                self.init_win_backward = tcp_window

        if fin:
            self.fin_count += 1

        if psh:
            self.psh_count += 1

        if ack:
            self.ack_count += 1

        self.last_time = timestamp

    # --------------------------------------------------
    # Utility functions
    # --------------------------------------------------

    @staticmethod
    def mean(values):
      return float(np.mean(values)) if len(values) > 0 else 0.0


    @staticmethod
    def std(values):
      return float(np.std(values)) if len(values) > 1 else 0.0


    @staticmethod
    def minimum(values):
      return float(np.min(values)) if len(values) > 0 else 0.0


    @staticmethod
    def maximum(values):
      return float(np.max(values)) if len(values) > 0 else 0.0

    # --------------------------------------------------
    # Generate 52 ML features
    # --------------------------------------------------

    def to_features(self):

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

        duration = max(
            self.last_time - self.start_time,
            1e-6
        )

        # ----------------------------------------------
        # Inter-arrival times
        # ----------------------------------------------

        all_timestamps = sorted(
            self.forward_timestamps +
            self.backward_timestamps
        )

        if len(all_timestamps) > 1:

            flow_iats = np.diff(all_timestamps)

        else:

            flow_iats = np.array([])

        fwd_iats = (
            np.diff(self.forward_timestamps)
            if len(self.forward_timestamps) > 1
            else np.array([])
        )

        bwd_iats = (
            np.diff(self.backward_timestamps)
            if len(self.backward_timestamps) > 1
            else np.array([])
        )

        # ----------------------------------------------
        # Packet statistics
        # ----------------------------------------------

        packet_lengths = all_packets

        # ----------------------------------------------
        # Features
        # ----------------------------------------------

        features = {

            "Destination Port":
                self.dst_port,

            "Flow Duration":
                duration,

            "Total Fwd Packets":
                total_fwd,

            "Total Length of Fwd Packets":
                total_fwd_bytes,

            "Fwd Packet Length Max":
                self.maximum(self.forward_packets),

            "Fwd Packet Length Min":
                self.minimum(self.forward_packets),

            "Fwd Packet Length Mean":
                self.mean(self.forward_packets),

            "Fwd Packet Length Std":
                self.std(self.forward_packets),

            "Bwd Packet Length Max":
                self.maximum(self.backward_packets),

            "Bwd Packet Length Min":
                self.minimum(self.backward_packets),

            "Bwd Packet Length Mean":
                self.mean(self.backward_packets),

            "Bwd Packet Length Std":
                self.std(self.backward_packets),

            "Flow Bytes/s":
                total_bytes / duration,

            "Flow Packets/s":
                total_packets / duration,

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
                if len(fwd_iats)
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
                if len(bwd_iats)
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
                sum(self.forward_header_lengths),

            "Bwd Header Length":
                sum(self.backward_header_lengths),

            "Fwd Packets/s":
                total_fwd / duration,

            "Bwd Packets/s":
                total_bwd / duration,

            "Min Packet Length":
                self.minimum(packet_lengths),

            "Max Packet Length":
                self.maximum(packet_lengths),

            "Packet Length Mean":
                self.mean(packet_lengths),

            "Packet Length Std":
                self.std(packet_lengths),

            "Packet Length Variance":
                float(np.var(packet_lengths))
                if packet_lengths
                else 0.0,

            "FIN Flag Count":
                self.fin_count,

            "PSH Flag Count":
                self.psh_count,

            "ACK Flag Count":
                self.ack_count,

            "Average Packet Size":
                total_bytes / total_packets
                if total_packets
                else 0.0,

            "Subflow Fwd Bytes":
                total_fwd_bytes,

            "Init_Win_bytes_forward":
                self.init_win_forward,

            "Init_Win_bytes_backward":
                self.init_win_backward,

            "act_data_pkt_fwd":
                total_fwd,

            "min_seg_size_forward":
                0,

            "Active Mean":
                self.mean(self.active_times),

            "Active Max":
                self.maximum(self.active_times),

            "Active Min":
                self.minimum(self.active_times),

            "Idle Mean":
                self.mean(self.idle_times),

            "Idle Max":
                self.maximum(self.idle_times),

            "Idle Min":
                self.minimum(self.idle_times),
        }

        return pd.DataFrame([features])
