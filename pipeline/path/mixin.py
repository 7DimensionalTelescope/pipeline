from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from collections import defaultdict
from typing import Union, List, Tuple, Iterable, Mapping, Sequence

from ..utils import add_suffix, swap_ext, collapse, atleast_1d


class AutoCollapseMixin:
    """Automatically collapses the output when it is a list of uniformly
    releated elemements"""

    # Define which attributes should be collapsed
    _collapse_exclude = {}  # "output_name", "name", "preprocess"}
    _collapse_include = {"masterframe"}  # "output_dir", "image_dir", "factory_dir", "stacked_dir"}

    def __getattribute__(self, name):
        # if name.startswith("_"):
        #     return object.__getattribute__(self, name)

        # value = object.__getattribute__(self, name)
        value = super().__getattribute__(name)

        if name.startswith("_"):
            return value

        # print("collapse", name, value)

        # Collapse if explicitly included or path-like list
        if name not in self._collapse_exclude and (
            name in self._collapse_include
            or (isinstance(value, list) and all(isinstance(v, (str, Path)) for v in value))
        ):
            # print("being collapsed", name, value)
            return collapse(value)

        return value

    # def __getattribute__(self, name):
    #     if name.startswith("_"):
    #         return object.__getattribute__(self, name)

    #     value = object.__getattribute__(self, name)
    #     # Only collapse specific attributes or path-like lists
    #     should_collapse = (
    #         (name in getattr(self, "_collapse_include", set()) and name not in self._collapse_exclude)
    #         or (isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value))
    #         or (isinstance(value, list) and "_yml" in name)
    #         or (isinstance(value, list) and "_dir" in name)
    #         or (isinstance(value, list) and "_log" in name)
    #     )

    #     if should_collapse:
    #         return collapse(value)

    #     return value


class AutoMkdirMixin:
    """This makes sure accessed dirs exist. Prepend _ to variables to prevent mkdir"""

    # attrs to skip mkdir; subclasses can modify this set
    _mkdir_exclude = {"output_name", "config_stem", "preproc_config_stem", "name", "changelog_dir"}

    def __init_subclass__(cls):
        # Ensure subclasses have their own created-directory cache
        cls._created_dirs_cache = set()

    def __getattribute__(self, name):
        """CAVEAT: This runs every time attr is accessed. Keep it short."""
        # if name.startswith("_"):  # Bypass all custom logic for private attributes
        #     return object.__getattribute__(self, name)

        # value = object.__getattribute__(self, name)

        value = super().__getattribute__(name)

        if name.startswith("_"):
            return value

        # print("mkdir", name)

        # Skip excluded attributes
        if name in object.__getattribute__(self, "_mkdir_exclude"):
            return value

        # print(f"AutoMkdirMixin debug: {name} {value}")

        # Handle vectorized paths
        if isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value):
            for p in value:
                self._mkdir(p)
        elif isinstance(value, (str, Path)):
            self._mkdir(value)

        # # DEBUG
        # for key in ["BIAS", "DARK", "FLAT"]:
        #     if isinstance(value, str) and key in os.path.dirname(value):
        #         print(f"AutoMkdirMixin _mkdir {name} {value}")
        #         raise RuntimeError(f"AutoMkdirMixin _mkdir {name} {value}")

        return value

    # def __getattr__(self, name):
    #     if name.startswith("_"):  # Bypass all custom logic for private attributes
    #         return object.__getattribute__(self, name)

    #     value = object.__getattribute__(self, name)

    #     if name in object.__getattribute__(self, "_mkdir_exclude"):
    #         return value

    #     if isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value):
    #         for p in value:
    #             self._mkdir(p)
    #     elif isinstance(value, (str, Path)):
    #         self._mkdir(value)

    #     return value

    def _mkdir(self, value):
        p = Path(value).expanduser()  # understands ~/
        d = p.parent if p.suffix else p  # ensure directory

        # Use mixin's own per-instance cache
        created_dirs = object.__getattribute__(self, "_created_dirs_cache")

        if d not in created_dirs and not d.exists():  # check cache first for performance
            # print(f"AutoMkdirMixin _mkdir creating directory: {d}")
            d.mkdir(parents=True, exist_ok=True)
            created_dirs.add(d)


# Under development
class MasterframeResolveMixin:
    """
    Adds resolve_masterframe()/require_masterframe() to PathPreprocess.

    - Never touches disk during normal attribute access.
    - Does a single os.path.exists check per unique path (cached).
    - On miss, builds a per-unit index {basename -> fullpath} under MASTER_FRAME_DIR/**/<unit>/
      and resolves by basename. Index is cached per process.
    """

    # Process-wide caches (safe enough for CLI/pipeline usage)
    _exists_cache: set[Path] = set()
    _resolve_cache: dict[str, str | None] = {}
    _index_by_unit: dict[str, dict[str, str]] = {}

    def _mf_exists(self, p: str | Path) -> bool:
        P = Path(p)
        if P in self._exists_cache:
            return True
        if P.exists():
            self._exists_cache.add(P)
            return True
        return False

    def _ensure_unit_index(self, unit: str):
        if unit in self._index_by_unit:
            return
        root = (
            Path(self._parent.const.MASTER_FRAME_DIR if hasattr(self._parent, "const") else None)
            or Path(self._parent.__class__.__module__.split(".")[0]).parent
        )  # fallback, optional
        root = Path(self._parent.__class__.__module__)  # placeholder if you inject const elsewhere

    @staticmethod
    @lru_cache(maxsize=None)
    def _build_unit_index(master_root: str, unit: str) -> dict[str, str]:
        """
        Walk MASTER_FRAME_DIR/**/<unit>/ and map basename -> absolute path.
        Assumes layout: MASTER_FRAME_DIR/<nightdate>/<unit>/**/<file>
        """
        mapping: dict[str, str] = {}
        unit = str(unit)
        root = Path(master_root)
        if not root.exists():
            return mapping
        # Visit only directories that contain this unit to keep it cheap
        for nd_dir in root.iterdir():
            udir = nd_dir / unit
            if not udir.is_dir():
                continue
            for dirpath, _, files in os.walk(udir):
                dp = Path(dirpath)
                for fn in files:
                    # Keep the last seen path (newer nights can override older if names collide)
                    mapping[fn] = str(dp / fn)
        return mapping

    def _resolve_one(self, p: str, unit: str, master_root: str) -> str | None:
        """Resolve a single masterframe path; returns full path or None."""
        if p in self._resolve_cache:
            return self._resolve_cache[p]

        if self._mf_exists(p):
            self._resolve_cache[p] = p
            return p

        idx = self._build_unit_index(master_root, unit)
        found = idx.get(os.path.basename(p))
        if found and self._mf_exists(found):
            self._resolve_cache[p] = found
            return found

        self._resolve_cache[p] = None
        return None

    # ---- public API for PathPreprocess ----
    def resolve_masterframe(self, *, strict: bool = False):
        """
        Returns the same shape as .masterframe, but with resolved paths (or original on miss).
        If strict=True, unresolved items become None (or raise via require_masterframe()).
        """
        # masterframe may be str, list[str], or list[list[str]] depending on type/science
        m = self.masterframe
        units = self._parent.name.unit  # vectorized
        master_root = self._parent.const.MASTER_FRAME_DIR if hasattr(self._parent, "const") else const.MASTER_FRAME_DIR

        def resolve_for(idx: int, path_or_list):
            u = units[idx if isinstance(units, list) else 0]
            if isinstance(path_or_list, list):
                out = []
                for p in path_or_list:
                    r = self._resolve_one(p, u, master_root)
                    out.append(r if (r or strict) else p)
                return out
            else:
                r = self._resolve_one(path_or_list, u, master_root)
                return r if (r or strict) else path_or_list

        if isinstance(m, list):
            # Vectorized over inputs
            return [resolve_for(i, item) for i, item in enumerate(m)]
        else:
            # Singleton
            return resolve_for(0, m)

    def require_masterframe(self):
        """
        Like resolve_masterframe(strict=True) but raises on any unresolved path.
        """
        resolved = self.resolve_masterframe(strict=True)

        def _check(x):
            if isinstance(x, list):
                for xi in x:
                    _check(xi)
            else:
                if x is None or not self._mf_exists(x):
                    raise FileNotFoundError(f"Masterframe not found: {x}")

        _check(resolved)
        return resolved
