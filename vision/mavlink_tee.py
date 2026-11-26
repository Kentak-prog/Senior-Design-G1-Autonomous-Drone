#!/usr/bin/env python3
"""
MAVLink tee on the VOXL 2: lets several programs share PX4's onboard MAVLink link at once.

voxl-vision-hub relays PX4's MAVLink to ONE localhost UDP port (en_localhost_mavlink_udp /
localhost_udp_port_number in /etc/modalai/voxl-vision-hub.conf, 14551 here), and only one program can
listen on a UDP port. This tee owns that port and copies every packet from PX4 to each client port;
whatever a client sends (commands, setpoints) goes back to PX4 through vision-hub unchanged.

    vision-hub ──udp──▶ 127.0.0.1:14551 (this tee) ──▶ 127.0.0.1:14601  flight script
                                                    └─▶ 127.0.0.1:14602  camera test (live_gate_test.py --mavlink)

Clients connect with pymavlink "udpin:127.0.0.1:<their port>": they receive from the tee and reply to
it, and the tee forwards their packets to vision-hub. Standard library only (Python 3.6+), so it runs
on the VOXL itself, outside Docker:

    nohup python3 /data/repo/vision/mavlink_tee.py > /data/mavlink_tee.log 2>&1 &

It must start before the clients. It prints packet counts every 10 s. If it stops, the clients stop
receiving PX4's position: the flight scripts treat that as stale position and stop, and PX4's own
OFFBOARD-loss failsafe (COM_OF_LOSS_T) takes over.
"""
import argparse
import select
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser(description="MAVLink UDP tee for PX4's localhost link on the VOXL 2")
    ap.add_argument("--listen", type=int, default=14551, help="port vision-hub sends PX4's MAVLink to (default 14551)")
    ap.add_argument("--clients", type=int, nargs="+", default=[14601, 14602],
                    help="client ports to copy PX4's packets to (default 14601 14602)")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", a.listen))
    except OSError as e:
        sys.exit("cannot listen on 127.0.0.1:%d (%s): is another program (a flight script or a second tee) "
                 "already using it?" % (a.listen, e))
    clients = [("127.0.0.1", p) for p in a.clients]
    upstream = None                     # vision-hub's address, learned from its first packet
    n_down = n_up = 0
    t_report = time.time()
    print("mavlink_tee: 127.0.0.1:%d -> %s" % (a.listen, ", ".join("%d" % p for p in a.clients)), flush=True)
    while True:
        r, _, _ = select.select([sock], [], [], 1.0)
        if r:
            try:
                data, addr = sock.recvfrom(65535)
            except OSError:                           # e.g. ICMP "port unreachable" for an absent client
                continue
            if addr in clients:                       # a client's command / setpoint -> PX4
                if upstream is not None:
                    sock.sendto(data, upstream)
                    n_up += 1
            else:                                     # PX4 (via vision-hub) -> every client
                if upstream != addr:
                    print("mavlink_tee: PX4 link from %s:%d" % addr, flush=True)
                upstream = addr
                for c in clients:
                    try:
                        sock.sendto(data, c)
                    except OSError:
                        pass                          # a client that isn't running yet is fine
                n_down += 1
        if time.time() - t_report >= 10:
            print("mavlink_tee: %d packets from PX4, %d to PX4 in the last 10 s" % (n_down, n_up), flush=True)
            n_down = n_up = 0
            t_report = time.time()


if __name__ == "__main__":
    main()
