"""Startup preflight shared by the stage, the run planner and the two claiming daemons, plus the systemd handshake."""

from __future__ import annotations

import os
import socket
import sys

from ..errors import Phot7DSError
from ..version import MIN_PHOT7DS_VERSION, is_below_min

STALE_UNIT_HINT = "unit is still Type=simple (no NOTIFY_SOCKET): re-install it from systemd/ and daemon-reload"


def check_phot7ds_version() -> str:
    """Installed phot7ds version; raises when it is below MIN_PHOT7DS_VERSION."""
    from phot7ds import __version__ as installed

    if is_below_min(installed, MIN_PHOT7DS_VERSION):
        raise Phot7DSError.PrerequisiteNotMetError(
            f"phot7ds {installed} is installed but >= {MIN_PHOT7DS_VERSION} is required"
        )
    return installed


def _unit_without_notify() -> bool:
    return bool(os.environ.get("INVOCATION_ID")) and not os.environ.get("NOTIFY_SOCKET")


def phot7ds_preflight(logger) -> str:
    """Installed phot7ds version; below the floor: one error line, then exit 78 (EX_CONFIG, never restarted)."""
    try:
        version = check_phot7ds_version()
    except (ImportError, Phot7DSError) as e:
        hint = f" ({STALE_UNIT_HINT})" if _unit_without_notify() else ""
        logger.error(f"Refusing to start: {e}. Upgrade phot7ds on this host, then restart.{hint}")
        sys.exit(os.EX_CONFIG)
    logger.info(f"phot7ds {version} satisfies the pipeline requirement")
    return version


def notify_ready(logger) -> None:
    """READY=1 to systemd for a Type=notify unit; warns when systemd started us without one, no-op outside systemd."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        if _unit_without_notify():
            logger.warning(STALE_UNIT_HINT)
        return
    if address.startswith("@"):
        address = "\0" + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.connect(address)
        sock.sendall(b"READY=1")
