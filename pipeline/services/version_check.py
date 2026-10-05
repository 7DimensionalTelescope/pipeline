from __future__ import annotations

from ..const.crossfilter import PHOT7DS_SPEC
from ..version import MIN_PHOT7DS_VERSION, MIN_SCIPROC_RUNTIME_VERSION, MIN_SCIPROC_RUNTIME_VERSION_MAP, is_below_min
from ..config.utils import get_key


def recorded_version(config_node, section: str) -> str | None:
    """Section's recorded runtime_version, falling back to info."""
    return get_key(getattr(config_node, section, None), "runtime_version") or get_key(
        config_node.info, "runtime_version"
    )


def floor_version(section: str) -> str:
    """Minimum runtime_version accepted for a config section."""
    return MIN_SCIPROC_RUNTIME_VERSION_MAP.get(section, MIN_SCIPROC_RUNTIME_VERSION)


def stale_reason(config_node, section: str) -> str | None:
    """Reason the section is stale, or None: runtime_version below floor, or phot7ds_version below the phot7ds floor."""
    recorded, minimum = recorded_version(config_node, section), floor_version(section)
    if is_below_min(recorded, minimum):
        return f"recorded runtime_version {recorded!r} below floor {minimum!r}"
    if section == PHOT7DS_SPEC.config_section:
        recorded = get_key(getattr(config_node, section, None), "phot7ds_version")
        if recorded and is_below_min(recorded, MIN_PHOT7DS_VERSION):  # no stamp means pre-stamp: runtime floor alone
            return f"recorded phot7ds_version {recorded!r} below floor {MIN_PHOT7DS_VERSION!r}"
    return None


def is_stale(config_node, section: str) -> bool:
    """True when the section's recorded runtime_version is missing or below its floor."""
    return stale_reason(config_node, section) is not None


class RuntimeVersionMixin:
    """Force overwrite=True when the last processed version is too old."""

    def resolve_overwrite(self, overwrite: bool) -> bool:
        if overwrite:
            return True

        spec = self._process_spec
        reason = stale_reason(self.config_node, spec.config_section)
        if reason:
            logger = getattr(self, "logger", None)
            msg = f"Escalating overwrite=True for {spec.name}: {reason} (section={spec.config_section!r})"
            if logger is not None:
                logger.info(msg)
            else:
                print(f"[INFO] {msg}")
            return True
        return False

    def record_runtime_version(self) -> None:
        """
        Stamp current pipeline version onto this process's config section.
        Use it after setting config.node.flag.x as it serves as overwrite indicator for successful runs.
        process_status DB captures the runtime version for failed runs.
        """
        from .. import __version__

        spec = self._process_spec
        section = getattr(self.config_node, spec.config_section, None)
        if section is not None:
            section.runtime_version = __version__
