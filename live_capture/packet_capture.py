import sys
import os
import signal
import time

# ============================================================
# Allow imports from project root
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from prediction.predict import predict_flow
from prediction.decision_engine import make_decision, print_decision

from scapy.all import sniff, IP, TCP, UDP

from feature_extractor import FlowStats


# ============================================================
# Configuration
# ============================================================

INTERFACE = "wlp0s20f3"

# Keep active flows in memory
flows = {}

# Remove inactive flows after this many seconds
FLOW_TIMEOUT = 15

# Used for graceful shutdown
STOP_CAPTURE = False


# ============================================================
# Well-known service ports
# ============================================================

WELL_KNOWN_PORTS = {
    20,      # FTP Data
    21,      # FTP
    22,      # SSH
    23,      # Telnet
    25,      # SMTP
    53,      # DNS
    67, 68,  # DHCP
    80,      # HTTP
    110,     # POP3
    123,     # NTP
    143,     # IMAP
    161,     # SNMP
    389,     # LDAP
    443,     # HTTPS
    445,     # SMB
    587,     # SMTP
    636,     # LDAPS
    993,     # IMAPS
    995,     # POP3S
    1433,    # MSSQL
    3306,    # MySQL
    3389,    # RDP
    5432,    # PostgreSQL
    8080,    # HTTP Alt
    8443     # HTTPS Alt
}


# ============================================================
# Flow identification
# ============================================================

def get_flow_key(packet):
    """
    Create a bidirectional flow key.

    Packets travelling in either direction belong
    to the same flow.
    """

    if IP not in packet:
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    # --------------------------------------------------------
    # TCP
    # --------------------------------------------------------

    if TCP in packet:

        protocol = "TCP"
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport

    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------

    elif UDP in packet:

        protocol = "UDP"
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    else:
        return None

    endpoint1 = (src_ip, src_port)
    endpoint2 = (dst_ip, dst_port)

    # --------------------------------------------------------
    # Sort endpoints so both directions use the same key
    # --------------------------------------------------------

    if endpoint1 <= endpoint2:

        return (
            protocol,
            endpoint1,
            endpoint2
        )

    else:

        return (
            protocol,
            endpoint2,
            endpoint1
        )


# ============================================================
# Packet processing
# ============================================================

def process_packet(packet):

    # Ignore packets without IP
    if IP not in packet:
        return

    flow_key = get_flow_key(packet)

    if flow_key is None:
        return

    protocol, endpoint1, endpoint2 = flow_key

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst

    # --------------------------------------------------------
    # Get source and destination ports
    # --------------------------------------------------------

    if TCP in packet:

        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport

    elif UDP in packet:

        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport

    else:

        return

    # --------------------------------------------------------
    # Determine packet direction
    # --------------------------------------------------------

    if (
        src_ip == endpoint1[0]
        and src_port == endpoint1[1]
    ):

        direction = "forward"

    else:

        direction = "backward"

    # --------------------------------------------------------
    # Packet information
    # --------------------------------------------------------

    timestamp = float(packet.time)

    packet_length = len(packet)

    # ========================================================
    # Create new flow
    # ========================================================

    if flow_key not in flows:

        # ----------------------------------------------------
        # Determine service / destination port
        # ----------------------------------------------------

        if src_port in WELL_KNOWN_PORTS:

            destination_port = src_port

        elif dst_port in WELL_KNOWN_PORTS:

            destination_port = dst_port

        else:

            destination_port = dst_port

        # ----------------------------------------------------
        # Create flow
        # ----------------------------------------------------

        flows[flow_key] = {

            "stats": FlowStats(
                destination_port,
                destination_port
            ),

            "last_seen": timestamp,

            # TCP termination state
            "rst": False,
            "fin_forward": False,
            "fin_backward": False
        }

        print(
            f"\n[+] New flow: "
            f"{src_ip}:{src_port} -> "
            f"{dst_ip}:{dst_port}"
        )

        print(
            f"    Service Port: {destination_port}"
        )

    # ========================================================
    # Get flow
    # ========================================================

    flow = flows[flow_key]

    # --------------------------------------------------------
    # Add packet statistics
    # --------------------------------------------------------

    flow["stats"].add_packet(
        packet_length,
        timestamp,
        direction,
        packet_length
    )

    flow["last_seen"] = timestamp

    # ========================================================
    # TCP flags
    # ========================================================

    if TCP in packet:

        flags = packet[TCP].flags

        # ----------------------------------------------------
        # RST flag
        # TCP RST = 0x04
        # ----------------------------------------------------

        if flags & 0x04:

            flow["rst"] = True

        # ----------------------------------------------------
        # FIN flag
        # TCP FIN = 0x01
        # ----------------------------------------------------

        if flags & 0x01:

            if direction == "forward":

                flow["fin_forward"] = True

            else:

                flow["fin_backward"] = True

        # ----------------------------------------------------
        # Update FlowStats counters
        # ----------------------------------------------------

        if hasattr(
            flow["stats"],
            "fin_flag_count"
        ):

            if flags & 0x01:

                flow["stats"].fin_flag_count += 1

        if hasattr(
            flow["stats"],
            "psh_flag_count"
        ):

            if flags & 0x08:

                flow["stats"].psh_flag_count += 1

        if hasattr(
            flow["stats"],
            "ack_flag_count"
        ):

            if flags & 0x10:

                flow["stats"].ack_flag_count += 1


# ============================================================
# Process completed flow
# ============================================================

def process_completed_flow(key, reason):

    flow = flows.pop(key, None)

    if flow is None:
        return

    try:

        features = flow["stats"].to_features()

        print(
            f"\n[FLOW COMPLETED - {reason}]"
        )

        print(
            f"Features extracted: "
            f"{len(features.columns)}"
        )

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        prediction = predict_flow(features)

        # ----------------------------------------------------
        # Decision engine
        # ----------------------------------------------------

        decision = make_decision(prediction)

        # ----------------------------------------------------
        # Display decision
        # ----------------------------------------------------

        print_decision(
            prediction,
            decision
        )

    except Exception as e:

        print(
            "\n[!] Error processing flow:"
        )

        print(e)


# ============================================================
# Remove completed / inactive flows
# ============================================================

def cleanup_flows():

    current_time = time.time()

    expired = []

    # --------------------------------------------------------
    # Check all active flows
    # --------------------------------------------------------

    for key, flow in list(flows.items()):

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

        # ----------------------------------------------------
        # TCP RST
        # ----------------------------------------------------

        if rst:

            expired.append(
                (key, "TCP RST")
            )

        # ----------------------------------------------------
        # Both directions sent FIN
        # ----------------------------------------------------

        elif (
            fin_forward
            and fin_backward
        ):

            expired.append(
                (key, "TCP FIN")
            )

        # ----------------------------------------------------
        # Inactive flow
        # ----------------------------------------------------

        elif (
            current_time
            - flow["last_seen"]
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

def stop_capture(signum, frame):

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

    print("=" * 60)

    print(
        "       SecureNet Live Packet Capture"
    )

    print("=" * 60)

    print(
        f"Interface    : {INTERFACE}"
    )

    print(
        f"Flow timeout : {FLOW_TIMEOUT} seconds"
    )

    print(
        "\nStarting packet capture..."
    )

    print(
        "Press CTRL+C to stop.\n"
    )

    # --------------------------------------------------------
    # Register signal handlers
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
    # Capture packets
    # ========================================================

    try:

        while not STOP_CAPTURE:

            try:

                # Capture packets for 2 seconds.
                #
                # The short timeout allows us to:
                #   1. Check flow timeouts
                #   2. Respond to Ctrl+C
                #   3. Keep the capture responsive
                #

                sniff(
                    iface=INTERFACE,
                    prn=process_packet,
                    store=False,
                    timeout=2
                )

            except Exception as e:

                print(
                    f"\n[!] Packet capture error: {e}"
                )

                # Don't immediately crash SecureNet.
                # Give Scapy a moment before retrying.

                if not STOP_CAPTURE:

                    time.sleep(1)

            # ------------------------------------------------
            # Check completed / inactive flows
            # ------------------------------------------------

            cleanup_flows()

    except KeyboardInterrupt:

        # This is a fallback in case Ctrl+C reaches here
        STOP_CAPTURE = True

    finally:

        print(
            "\n[!] Final cleanup..."
        )

        # ====================================================
        # Process remaining flows
        # ====================================================

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

        # ----------------------------------------------------
        # Clear flow table
        # ----------------------------------------------------

        flows.clear()

        print(
            "\n[+] SecureNet capture stopped."
        )