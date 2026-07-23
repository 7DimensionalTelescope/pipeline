from __future__ import annotations

from ..version import MIN_SCIPROC_RUNTIME_VERSION, MIN_SCIPROC_RUNTIME_VERSION_MAP, is_below_min
from ..config.utils import get_key


class RuntimeVersionMixin:
    """Force overwrite=True when the last processed version is too old."""

    def resolve_overwrite(self, overwrite: bool) -> bool:
        if overwrite:
            return True

        spec = self._process_spec
        section = getattr(self.config_node, spec.config_section, None)
        recorded = get_key(section, "runtime_version") or get_key(self.config_node.info, "runtime_version")
        minimum = MIN_SCIPROC_RUNTIME_VERSION_MAP.get(spec.config_section, MIN_SCIPROC_RUNTIME_VERSION)

        if is_below_min(recorded, minimum):
            logger = getattr(self, "logger", None)
            msg = (
                f"Escalating overwrite=True for {spec.name}: "
                f"recorded runtime_version={recorded!r} < minimum={minimum!r} (section={spec.config_section!r})"
            )
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
