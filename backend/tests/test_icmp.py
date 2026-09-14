"""Parser tests against captured ping output from real implementations."""

import pytest

from backend.app.collectors.icmp import parse_ping_output

IPUTILS_HEALTHY = """PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 ttl=118 time=12.3 ms
64 bytes from 8.8.8.8: icmp_seq=2 ttl=118 time=12.5 ms

--- 8.8.8.8 ping statistics ---
5 packets transmitted, 5 received, 0% packet loss, time 4006ms
rtt min/avg/max/mdev = 12.301/12.456/12.678/0.134 ms
"""

IPUTILS_PARTIAL_LOSS = """--- 10.0.0.5 ping statistics ---
5 packets transmitted, 3 received, 40% packet loss, time 4052ms
rtt min/avg/max/mdev = 45.100/61.250/89.400/18.220 ms
"""

IPUTILS_TOTAL_LOSS = """PING 192.0.2.1 (192.0.2.1) 56(84) bytes of data.

--- 192.0.2.1 ping statistics ---
5 packets transmitted, 0 received, 100% packet loss, time 4104ms
"""

BSD_HEALTHY = """PING 8.8.8.8 (8.8.8.8): 56 data bytes
64 bytes from 8.8.8.8: icmp_seq=0 ttl=118 time=12.301 ms

--- 8.8.8.8 ping statistics ---
5 packets transmitted, 5 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 12.301/12.456/12.678/0.134 ms
"""


def test_healthy_link_iputils():
    result = parse_ping_output(IPUTILS_HEALTHY)
    assert result.packet_loss_pct == 0.0
    assert result.latency_ms == pytest.approx(12.456)
    assert result.jitter_ms == pytest.approx(0.134)
    assert result.reachable


def test_partial_loss_keeps_both_loss_and_timing():
    result = parse_ping_output(IPUTILS_PARTIAL_LOSS)
    assert result.packet_loss_pct == 40.0
    assert result.latency_ms == pytest.approx(61.25)
    assert result.jitter_ms == pytest.approx(18.22)
    assert result.reachable


def test_total_loss_is_a_result_not_a_parse_failure():
    """An unreachable device is the signal we most want, not an error."""
    result = parse_ping_output(IPUTILS_TOTAL_LOSS)
    assert result.packet_loss_pct == 100.0
    assert result.latency_ms is None
    assert result.jitter_ms is None
    assert not result.reachable


def test_bsd_format_with_decimal_loss_and_stddev():
    result = parse_ping_output(BSD_HEALTHY)
    assert result.packet_loss_pct == 0.0
    assert result.latency_ms == pytest.approx(12.456)
    assert result.jitter_ms == pytest.approx(0.134)


def test_unrecognisable_output_returns_none():
    assert parse_ping_output("ping: unknown host nope.invalid") is None
    assert parse_ping_output("") is None
