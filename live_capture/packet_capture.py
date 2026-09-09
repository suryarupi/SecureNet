import sys
import os
import signal
import time

# ============================================================
# Allow imports from project root
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from prediction.predict import predict_flow
from prediction.decision_engine import (
    make_decision,
    print_decision
)

# ------------------------------------------------------------
# NEW: live data store (SQLite bridge to the dashboard)
# ------------------------------------------------------------
from dashboard.live_data import LiveDataStore

from scapy.all import (
    sniff,
    IP,
    TCP,
    UDP
)

from feature_extractor import FlowStats


# ============================================================
# Configuration
# ============================================================

INTERFACE = "wlp0s20f3"

FLOW_TIMEOUT = 15

# Minimum number of packets required before
# sending a flow to the ML models.
MIN_PACKETS = 2

STOP_CAPTURE = False

flows = {}

# ------------------------------------------------------------
# NEW: single long-lived store used for the whole capture run
# ------------------------------------------------------------
store = LiveDataStore()


# ============================================================
# Well-known service ports
# ============================================================

WELL_KNOWN_PORTS = {
    20, 21, 22, 23, 25,
    53,
    67, 68,
    80,
    110,
    123,
    143,
    161,
    389,
    443,
    445,
    587,
    636,
    993,
    995,
    1433,
    3306,
    3389,
    5432,
    5222,
    8080,
    8443
}


# ============================================================
# Get packet transport information
# ============================================================

def get_transport_info(packet):

    if TCP in packet:

        return (
            "TCP",
            int(packet[TCP].sport),
            int(packet[TCP].dport)
        )

    if UDP in packet:

        return (
            "UDP",
            int(packet[UDP].sport),
            int(packet[UDP].dport)
        )

    return None


# ============================================================
# Create bidirectional flow key
# ============================================================

def get_flow_key(packet):

    if IP not in packet:
        return None

    transport = get_transport_info(packet)

    if transport is None:
        return None

    protocol, src_port, dst_port = transport

    src_endpoint = (
        packet[IP].src,
        src_port
    )

    dst_endpoint = (
        packet[IP].dst,
        dst_port
    )

    # --------------------------------------------------------
    # Sort endpoints so both directions belong to
    # exactly the same flow.
    # --------------------------------------------------------

    if src_endpoint <= dst_endpoint:

        first = src_endpoint
        second = dst_endpoint

    else:

        first = dst_endpoint
        second = src_endpoint

    return (
        protocol,
        first,
        second
    )


# ============================================================
# Determine direction
# ============================================================

def get_direction(packet, flow_key):

    _, endpoint1, endpoint2 = flow_key

    transport = get_transport_info(packet)

    if transport is None:
        return None

    _, src_port, _ = transport

    src_endpoint = (
        packet[IP].src,
        src_port
    )

    if src_endpoint == endpoint1:

        return "forward"

    if src_endpoint == endpoint2:

        return "backward"

    return None


# ============================================================
# Determine service / destination port
# ============================================================

def get_service_port(src_port, dst_port):

    # --------------------------------------------------------
    # If destination is a well-known service port,
    # it is the service port.
    # --------------------------------------------------------

    if dst_port in WELL_KNOWN_PORTS:

        return dst_port

    # --------------------------------------------------------
    # If source is a well-known service port,
    # this packet is probably travelling from server
    # to client.
    # --------------------------------------------------------

    if src_port in WELL_KNOWN_PORTS:

        return src_port

    # --------------------------------------------------------
    # Otherwise use destination port.
    # --------------------------------------------------------

    return dst_port


# ============================================================
# Extract packet metadata
# ============================================================

def extract_packet_metadata(packet):

    packet_length = len(packet)

    header_length = 0
    tcp_window = 0

    fin = False
    psh = False
    ack = False
    rst = False

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    if TCP in packet:

        tcp = packet[TCP]

        # TCP header length
        if tcp.dataofs is not None:

            header_length = int(
                tcp.dataofs
            ) * 4

        # TCP receive window
        tcp_window = int(
            tcp.window
        )

        flags = int(
            tcp.flags
        )

        fin = bool(
            flags & 0x01
        )

        rst = bool(
            flags & 0x04
        )

        psh = bool(
            flags & 0x08
        )

        ack = bool(
            flags & 0x10
        )

    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------

    elif UDP in packet:

        # UDP header = 8 bytes
        header_length = 8

    return {
        "packet_length": packet_length,
        "header_length": header_length,
        "tcp_window": tcp_window,
        "fin": fin,
        "psh": psh,
        "ack": ack,
        "rst": rst
    }


# ============================================================
# Process packet
# ============================================================

def process_packet(packet):

    # --------------------------------------------------------
    # Only process IPv4 packets
    # --------------------------------------------------------

    if IP not in packet:
        return

    transport = get_transport_info(packet)

    if transport is None:
        return

    protocol, src_port, dst_port = transport

    # --------------------------------------------------------
    # Flow key
    # --------------------------------------------------------

    flow_key = get_flow_key(packet)

    if flow_key is None:
        return

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction = get_direction(
        packet,
        flow_key
    )

    if direction is None:
        return

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = float(
        packet.time
    )

    # --------------------------------------------------------
    # Create new flow
    # --------------------------------------------------------

    if flow_key not in flows:

        service_port = get_service_port(
            src_port,
            dst_port
        )

        flows[flow_key] = {

            "stats": FlowStats(
                service_port
            ),

            "last_seen": timestamp,

            "packet_count": 0,

            "rst": False,

            "fin_forward": False,

            "fin_backward": False,

            "protocol": protocol,

            # ------------------------------------------------
            # NEW: remember the originating endpoints so the
            # dashboard can display readable src/dst columns.
            # This does NOT affect ML features - FlowStats
            # already tracks direction independently via
            # get_direction() above.
            # ------------------------------------------------
            "src_ip": packet[IP].src,
            "src_port": src_port,
            "dst_ip": packet[IP].dst,
            "dst_port": dst_port
        }

        print(
            f"\n[+] New flow: "
            f"{packet[IP].src}:{src_port} -> "
            f"{packet[IP].dst}:{dst_port}"
        )

        print(
            f"    Protocol    : {protocol}"
        )

        print(
            f"    Service Port: {service_port}"
        )

    # --------------------------------------------------------
    # Get flow
    # --------------------------------------------------------

    flow = flows[flow_key]

    # --------------------------------------------------------
    # Extract packet metadata
    # --------------------------------------------------------

    metadata = extract_packet_metadata(
        packet
    )

    # --------------------------------------------------------
    # Add packet statistics
    # --------------------------------------------------------

    flow["stats"].add_packet(

        packet_length=metadata[
            "packet_length"
        ],

        timestamp=timestamp,

        direction=direction,

        header_length=metadata[
            "header_length"
        ],

        tcp_window=metadata[
            "tcp_window"
        ],

        fin=metadata[
            "fin"
        ],

        psh=metadata[
            "psh"
        ],

        ack=metadata[
            "ack"
        ]
    )

    # --------------------------------------------------------
    # Update flow state
    # --------------------------------------------------------

    flow["last_seen"] = timestamp

    flow["packet_count"] += 1

    # --------------------------------------------------------
    # TCP RST
    # --------------------------------------------------------

    if metadata["rst"]:

        flow["rst"] = True

    # --------------------------------------------------------
    # TCP FIN
    # --------------------------------------------------------

    if metadata["fin"]:

        if direction == "forward":

            flow["fin_forward"] = True

        else:

            flow["fin_backward"] = True


# ============================================================
# Process completed flow
# ============================================================

def process_completed_flow(
    key,
    reason
):

    flow = flows.pop(
        key,
        None
    )

    if flow is None:
        return

    packet_count = flow[
        "packet_count"
    ]

    # --------------------------------------------------------
    # Ignore extremely small flows
    # --------------------------------------------------------

    if packet_count < MIN_PACKETS:

        print(
            f"\n[FLOW IGNORED - {reason}]"
        )

        print(
            f"    Packets: {packet_count}"
        )

        print(
            "    Reason: "
            "Insufficient packets for flow classification."
        )

        return

    try:

        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = flow[
            "stats"
        ].to_features()

        print(
            f"\n[FLOW COMPLETED - {reason}]"
        )

        print(
            "\n--- LIVE FEATURES ---"
        )

        print(
            features.T.to_string()
        )

        print(
            f"Features extracted: "
            f"{len(features.columns)}"
        )

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        prediction = predict_flow(
            features
        )

        # ----------------------------------------------------
        # Decision engine
        # ----------------------------------------------------

        decision = make_decision(
            prediction
        )

        # ----------------------------------------------------
        # Display decision
        # ----------------------------------------------------

        print_decision(
            prediction,
            decision
        )

        # ----------------------------------------------------
        # NEW: persist the result for the dashboard.
        # Wrapped in its own try/except so a storage problem
        # can never take down the live pipeline above.
        # ----------------------------------------------------

        try:

            row = {
                "timestamp": time.time(),
                "src_ip": flow.get("src_ip"),
                "src_port": flow.get("src_port"),
                "dst_ip": flow.get("dst_ip"),
                "dst_port": flow.get("dst_port"),
                "protocol": flow.get("protocol"),
                "status": decision["status"],
                "severity": decision["severity"],
                "attack_type": decision["attack_type"],
                "attack_probability": prediction["attack_probability"],
                "attack_confidence": prediction["attack_confidence"],
                "anomaly": prediction["anomaly"],
                "anomaly_score": prediction["anomaly_score"],
                "reason": decision["reason"],
            }

            try:
                features_dict = features.iloc[0].to_dict()
            except Exception:
                features_dict = None

            store.insert_flow(row, features=features_dict)

        except Exception as store_error:

            print(
                "\n[!] Error storing flow for dashboard "
                f"(pipeline unaffected): {store_error}"
            )

    except Exception as e:

        print(
            "\n[!] Error processing flow:"
        )

        print(
            f"    {type(e).__name__}: {e}"
        )


# ============================================================
# Cleanup inactive / completed flows
# ============================================================

def cleanup_flows():

    current_time = time.time()

    expired = []

    # --------------------------------------------------------
    # Examine active flows
    # --------------------------------------------------------

    for key, flow in list(
        flows.items()
    ):

        rst = flow.get(
            "rst",
            False
        )

        fin_forward = flow.get(
            "fin_forward",
            False
        )

        fin_backward = flow.get(
            "fin_backward",
            False
        )

        last_seen = flow.get(
            "last_seen",
            current_time
        )

        # ----------------------------------------------------
        # RST
        # ----------------------------------------------------

        if rst:

            expired.append(
                (key, "TCP RST")
            )

        # ----------------------------------------------------
        # Both sides FIN
        # ----------------------------------------------------

        elif (
            fin_forward
            and fin_backward
        ):

            expired.append(
                (key, "TCP FIN")
            )

        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        elif (
            current_time -
            last_seen
            > FLOW_TIMEOUT
        ):

            expired.append(
                (key, "TIMEOUT")
            )

    # --------------------------------------------------------
    # Process completed flows
    # --------------------------------------------------------

    for key, reason in expired:

        process_completed_flow(
            key,
            reason
        )


# ============================================================
# Signal handler
# ============================================================

def stop_capture(
    signum,
    frame
):

    global STOP_CAPTURE

    if STOP_CAPTURE:
        return

    print(
        "\n\n[!] Stop signal received."
    )

    print(
        "[!] Stopping packet capture..."
    )

    STOP_CAPTURE = True


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "       SecureNet Live Packet Capture"
    )

    print(
        "=" * 60
    )

    print(
        f"Interface       : {INTERFACE}"
    )

    print(
        f"Flow timeout    : {FLOW_TIMEOUT} seconds"
    )

    print(
        f"Minimum packets : {MIN_PACKETS}"
    )

    print(
        "\nStarting packet capture..."
    )

    print(
        "Press CTRL+C to stop.\n"
    )

    # --------------------------------------------------------
    # Signal handlers
    # --------------------------------------------------------

    signal.signal(
        signal.SIGINT,
        stop_capture
    )

    signal.signal(
        signal.SIGTERM,
        stop_capture
    )

    # ========================================================
    # Capture loop
    # ========================================================

    try:

        while not STOP_CAPTURE:

            try:

                sniff(
                    iface=INTERFACE,
                    prn=process_packet,
                    store=False,
                    timeout=2
                )

            except Exception as e:

                print(
                    f"\n[!] Packet capture error: "
                    f"{e}"
                )

                if not STOP_CAPTURE:

                    time.sleep(1)

            # ------------------------------------------------
            # Check flow completion
            # ------------------------------------------------

            cleanup_flows()

            # ------------------------------------------------
            # NEW: heartbeat so the dashboard knows capture
            # is alive, even during quiet traffic periods.
            # ------------------------------------------------

            store.heartbeat(INTERFACE)

    except KeyboardInterrupt:

        STOP_CAPTURE = True

    finally:

        print(
            "\n[!] Final cleanup..."
        )

        remaining_flows = list(
            flows.keys()
        )

        if remaining_flows:

            print(
                f"[!] Processing "
                f"{len(remaining_flows)} "
                f"remaining flow(s)..."
            )

            for key in remaining_flows:

                process_completed_flow(
                    key,
                    "CAPTURE STOPPED"
                )

        flows.clear()

        # ------------------------------------------------
        # NEW: close the DB connection cleanly.
        # ------------------------------------------------
        store.close()

        print(
            "\n[+] SecureNet capture stopped."
        )