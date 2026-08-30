"""Selective re-preprocessing: ingredient staleness detection and manual master designation.

Semantics, rationale and measurements: .claude/memory/processing-stages.md and api-recipes.md.
"""

import os
from functools import lru_cache

from astropy.io import fits

from ..config.utils import get_key
from ..const import (
    CALIB_TYPE_BIAS,
    CALIB_TYPE_DARK,
    CALIB_TYPE_FLAT,
    CALIB_TYPES,
    NAME_TYPE_BIAS,
    NAME_TYPE_DARK,
    NAME_TYPE_FLAT,
    NAME_TYPE_MASTER,
)
from ..errors import PreprocessError
from ..path.name import NameHandler
from .utils import get_image_id


def recorded_ingredient_ids(image) -> dict:
    """{ingredient basename: IMAGEID or None} from the IMCMB/IMCID cards; None = unidentified."""
    # from the FITS, never the .header sidecar: the sidecar is staged ahead of the pixels
    try:
        header = fits.getheader(image)
    except Exception:
        return {}

    recorded = {}
    for key in header:
        if not key.startswith("IMCMB"):
            continue
        name = str(header[key] or "").strip()
        image_id = str(header.get(f"IMCID{key[5:]}", "") or "").strip()
        if name:
            recorded[os.path.basename(name)] = image_id or None
    return recorded


@lru_cache(maxsize=4096)
def _resolve_master_path(basename: str, is_pipeline: bool = True) -> str | None:
    from ..path.path import PathHandler
    from ..utils import atleast_1d

    path = PathHandler(basename, is_pipeline=is_pipeline)
    if atleast_1d(path.name.type)[0].kind != NAME_TYPE_MASTER:
        return None
    return atleast_1d(path._resolved_files)[0]


def resolve_master_path(basename: str, is_pipeline: bool = True) -> str | None:
    """On-disk path of a master frame from its basename alone; None if it is not a master."""
    # failures are not memoized: a transient NFS error must not pin None for the process lifetime
    try:
        return _resolve_master_path(basename, is_pipeline)
    except Exception:
        return None


def ingredient_state(path: str, cache: dict = None) -> tuple:
    """(IMAGEID, SANITY) of an ingredient as it stands on disk."""
    if cache is not None and path in cache:
        return cache[path]
    try:
        sanity = fits.getval(path, "SANITY")
    except Exception:
        sanity = None
    state = (get_image_id(path), sanity)
    if cache is not None:
        cache[path] = state
    return state


def ingredient_change(
    image,
    selected=(),
    better_match: bool = False,
    is_pipeline: bool = True,
    cache: dict = None,
    designated=frozenset(),
):
    """Why `image` must be rebuilt: "designated" | "regenerated" | "sanity" | "better-match" | None."""
    from .ppflag import PPFLAG_SANITY_F_USED, get_ppflag_from_header

    recorded = recorded_ingredient_ids(image)

    # explicit user intent outranks the unknown-provenance guard below
    for name in designated:
        if name not in recorded:
            return "designated"

    if not recorded:
        return None

    used_rejected = False
    sanity_known = True
    for name, was in recorded.items():
        path = resolve_master_path(name, is_pipeline)
        if path is None:
            continue
        if was is None:
            # used but unidentified: regeneration is invisible, and the sanity test would churn forever
            sanity_known = False
            continue
        image_id, sanity = ingredient_state(path, cache=cache)
        if image_id and image_id != was:
            return "regenerated"
        if sanity is None:
            sanity_known = False
        elif sanity is False:
            used_rejected = True

    if sanity_known:
        try:
            recorded_flag = bool(get_ppflag_from_header(image, raise_if_missing=True) & PPFLAG_SANITY_F_USED)
            if recorded_flag != used_rejected:
                return "sanity"
        except Exception:
            pass

    if better_match:
        for ingredient in selected:
            if ingredient and os.path.basename(ingredient) not in recorded:
                return "better-match"
    return None


class ReprocessMixin:
    """Preprocess methods for staleness decisions and designated master frames."""

    def _ingredient_change(self, product, selected):
        """Why `product` is out of date, or None."""
        return ingredient_change(
            product,
            selected,
            better_match=self._better_match,
            is_pipeline=self._is_pipeline,
            cache=self._ingredient_cache,
            designated=self._designated_basenames.intersection(os.path.basename(p) for p in selected if p),
        )

    def _master_change(self, dtype, output_file):
        """Why an existing master must be rebuilt, or None."""
        if dtype == CALIB_TYPE_DARK:
            selected = (self.bias_output,)
        elif dtype == CALIB_TYPE_FLAT:
            selected = (self.bias_output, self.flatdark_output)
        else:
            return None  # a bias is combined from raw frames alone

        reason = self._ingredient_change(output_file, selected)
        if reason:
            self.logger.info(
                f"[Group {self._current_group+1}] Master {dtype} is out of date ({reason}); "
                f"regenerating {os.path.basename(output_file)}"
            )
        return reason

    def _sci_change(self, output_file):
        """Why a science output must be (re)produced, or None."""
        if not os.path.exists(output_file):
            return "missing"
        return self._ingredient_change(output_file, (self.bias_output, self.dark_output, self.flat_output))

    _ROLE_DTYPE = {
        "bias": CALIB_TYPE_BIAS,
        "dark": CALIB_TYPE_DARK,
        "flat": CALIB_TYPE_FLAT,
        "flatdark": CALIB_TYPE_DARK,
    }

    def _load_designated_masterframes(self):
        """Validate `preprocess.designated_masterframes` into {dtype: [(path, NameHandler, role)]}.

        An entry is a path (serves every role its identity keys match) or a one-key
        mapping `role: path` pinning a single role (bias/dark/flat/flatdark).
        """
        self._designated = {}
        self._designated_basenames = frozenset()
        self._designation_dispatch = {}

        entries = get_key(self.config_node.preprocess, "designated_masterframes", None) or []
        if isinstance(entries, (str, dict)):
            entries = [entries]
        for entry in entries:
            role = None
            path = entry
            if isinstance(entry, dict):
                if len(entry) != 1 or next(iter(entry)) not in self._ROLE_DTYPE:
                    raise PreprocessError.ValueError(
                        f"designated_masterframes: an entry must be a path or one `role: path` "
                        f"mapping with role in {sorted(self._ROLE_DTYPE)}: {entry}"
                    )
                role, path = next(iter(entry.items()))
            if not os.path.exists(path):
                raise PreprocessError.ValueError(f"designated_masterframes: file not found: {path}")
            try:
                name = NameHandler(path)
                dtype = name.type.exposure_type if name.type.kind == NAME_TYPE_MASTER else None
            except Exception:
                dtype = None
            if dtype not in (NAME_TYPE_BIAS, NAME_TYPE_DARK, NAME_TYPE_FLAT):
                raise PreprocessError.ValueError(f"designated_masterframes: not a master bias/dark/flat: {path}")
            if role is not None and self._ROLE_DTYPE[role] != dtype:
                raise PreprocessError.ValueError(
                    f"designated_masterframes: role {role!r} needs a master "
                    f"{self._ROLE_DTYPE[role]}, got a {dtype}: {path}"
                )
            if not get_image_id(path):
                # an id-less designation could never converge; overwrite=True is the tool there
                raise PreprocessError.ValueError(f"designated_masterframes: no IMAGEID in {path}")
            self._designated.setdefault(dtype, []).append((path, name, role))
        if self._designated:
            self._designated_basenames = frozenset(
                os.path.basename(p) for lst in self._designated.values() for p, _, _ in lst
            )
            self._change_policy += "+designated"

    def _designated_for(self, dtype, template, group_index):
        """The designated master for one group's `dtype` role, or None."""
        if not self._designated.get(dtype) or not template:
            return None
        target = NameHandler(template)
        matches = []
        for path, name, role in self._designated[dtype]:
            if role is not None and role != dtype:
                continue
            if str(name.n_binning) != str(target.n_binning):
                continue
            if dtype == CALIB_TYPE_FLAT and name.filter != target.filter:
                continue
            if dtype == CALIB_TYPE_DARK and str(name.exptime) != str(target.exptime):
                continue  # the reduction kernel subtracts the dark unscaled
            matches.append((path, name))
        if not matches:
            return None
        if len(matches) > 1:
            raise PreprocessError.ValueError(
                f"designated_masterframes: {len(matches)} candidates for the {dtype} role of "
                f"group {group_index+1} ({os.path.basename(template)}): "
                f"{[os.path.basename(p) for p, _ in matches]}"
            )
        path, name = matches[0]
        for key in ("unit", "gain", "camera"):
            if str(getattr(name, key)) != str(getattr(target, key)):
                self.logger.warning(
                    f"[Group {group_index+1}] Designated {dtype} {os.path.basename(path)} "
                    f"differs from the group in {key} ({getattr(name, key)} != {getattr(target, key)})"
                )
        return path

    def _dispatch_designated_masterframes(self):
        """Resolve and log {(group index, role): designated path} for every group, up front."""
        self._designation_dispatch = {}
        if not self._designated:
            return

        for i in range(self._n_groups):
            templates = self.raw_groups[i][1]
            for dtype in CALIB_TYPES:
                path = self._designated_for(dtype, templates[self._key_to_index[dtype]], i)
                if path:
                    self._designation_dispatch[(i, dtype)] = path

            dark_template = templates[self._key_to_index[CALIB_TYPE_DARK]]
            if dark_template and self._designated.get(CALIB_TYPE_DARK):
                # flatdark role is exptime-free; an explicit `flatdark:` pin outranks plain darks (min exptime)
                target = NameHandler(dark_template)
                candidates = [
                    (name.exptime, path, role)
                    for path, name, role in self._designated[CALIB_TYPE_DARK]
                    if role in (None, "flatdark") and str(name.n_binning) == str(target.n_binning)
                ]
                pinned = [c for c in candidates if c[2] == "flatdark"]
                if len(pinned) > 1:
                    raise PreprocessError.ValueError(
                        f"designated_masterframes: {len(pinned)} flatdark pins match group {i+1}: "
                        f"{[os.path.basename(p) for _, p, _ in pinned]}"
                    )
                if pinned:
                    self._designation_dispatch[(i, "flatdark")] = pinned[0][1]
                elif candidates:
                    self._designation_dispatch[(i, "flatdark")] = min(candidates)[1]

        for (i, role), path in sorted(self._designation_dispatch.items()):
            if role in self._key_to_index:
                template = self.raw_groups[i][1][self._key_to_index[role]]
                where = f" (in place of {os.path.basename(template)})"
            else:
                where = ""
            self.logger.info(f"[Group {i+1}] Designated {role}: {os.path.basename(path)}{where}")

        unmatched = {p for lst in self._designated.values() for p, _, _ in lst} - set(
            self._designation_dispatch.values()
        )
        if unmatched:
            raise PreprocessError.ValueError(
                "designated_masterframes: no group role matches "
                + ", ".join(sorted(os.path.basename(p) for p in unmatched))
                + " -- check filter/exptime/binning against this run's groups"
            )

    def _adopt_designated_masterframe(self, designated_file, dtype, template):
        """Use a designated master for this group: same bookkeeping as a fetch."""
        from . import ppflag

        try:
            sanity = fits.getval(designated_file, "SANITY")
        except Exception:
            sanity = None
            self.logger.warning(
                f"[Group {self._current_group+1}] Designated {dtype} has no SANITY card: {designated_file}"
            )
        self.logger.info(
            f"[Group {self._current_group+1}] Using designated (sanity: {sanity}) master {dtype} "
            f"at {os.path.basename(designated_file)} in place of {os.path.basename(template)}"
        )
        self._ppflag[dtype] = ppflag.compute_fetch_ppflag(designated_file, template, sanity)
        self.raw_groups[self._current_group][1][self._key_to_index[dtype]] = designated_file
