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
from ..const.observation import BIAS_GROUP_LENIENT_KEYS, DARK_GROUP_LENIENT_KEYS, FLAT_GROUP_LENIENT_KEYS
from ..errors import PreprocessError
from ..path.name import NameHandler
from .utils import get_image_id


def _stem(path) -> str:
    return os.path.basename(str(path)).removesuffix(".fits")


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
    _LENIENT_KEYS = {
        CALIB_TYPE_BIAS: BIAS_GROUP_LENIENT_KEYS,
        CALIB_TYPE_DARK: DARK_GROUP_LENIENT_KEYS,
        CALIB_TYPE_FLAT: FLAT_GROUP_LENIENT_KEYS,
    }

    def _load_designated_masterframes(self):
        """Validate `preprocess.designated_masterframes` into {dtype: [(path, NameHandler, role, image stems)]}."""
        self._designated = {}
        self._designated_basenames = frozenset()
        self._designation_dispatch = {}
        self._image_pins = {}
        self._reuse_recorded = bool(get_key(self.config_node.preprocess, "reuse_recorded_masters", False))
        if self._reuse_recorded:
            self._change_policy += "+recorded"

        entries = get_key(self.config_node.preprocess, "designated_masterframes", None) or []
        if isinstance(entries, (str, dict)):
            entries = [entries]
        for entry in entries:
            role = None
            images = ()
            path = entry
            if isinstance(entry, dict):
                images = entry.get("images") or ()
                images = tuple(_stem(s) for s in ([images] if isinstance(images, str) else images))
                keys = [k for k in entry if k != "images"]
                if len(keys) != 1 or keys[0] not in self._ROLE_DTYPE or (images and keys[0] == "flatdark"):
                    raise PreprocessError.ValueError(
                        f"designated_masterframes: an entry must be a path or one `role: path` mapping with role in "
                        f"{sorted(self._ROLE_DTYPE)}, optionally with `images:` (not for flatdark): {entry}"
                    )
                role, path = keys[0], entry[keys[0]]
            if not os.path.exists(path):
                path = resolve_master_path(os.path.basename(str(path)), self._is_pipeline) or path
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
            self._designated.setdefault(dtype, []).append((path, name, role, images))
        if self._designated:
            self._designated_basenames = frozenset(
                os.path.basename(p) for lst in self._designated.values() for p, _, _, _ in lst
            )
            self._change_policy += "+designated"

    def _add_recorded_pins(self, pins_of):
        """Pin each frame's roles to the masters its own IMCMB cards record where they differ from the group's templates."""
        frames = pinned = unusable = 0
        for i, (_, masters, sci_dict) in enumerate(self.raw_groups):
            # an explicit designation outranks the record: group-level (a designated flatdark shapes the flat), or a pin
            explicit = {d for d in CALIB_TYPES if self._designated_for(d, masters[self._key_to_index[d]], i)}
            if any(r in (None, "flatdark") and not imgs for _, _, r, imgs in self._designated.get(CALIB_TYPE_DARK, [])):
                explicit.add(CALIB_TYPE_FLAT)
            for raw, processed in (pair for pairs in sci_dict.values() for pair in zip(*pairs)):
                record = recorded_ingredient_ids(processed)
                if not record:
                    continue
                frames += 1
                frame = None
                for basename in record:
                    kind = NameHandler(basename).type
                    dtype = kind.exposure_type if kind.kind == NAME_TYPE_MASTER else None
                    if dtype not in self._key_to_index:
                        continue
                    template = masters[self._key_to_index[dtype]]
                    taken = dtype in explicit or any(
                        dtype in pins_of.get(s, {}) for s in (_stem(raw), _stem(processed))
                    )
                    if taken or (template and os.path.basename(template) == basename):
                        continue
                    path = resolve_master_path(basename, self._is_pipeline)
                    frame = frame or NameHandler(raw)
                    master = NameHandler(path) if path and os.path.exists(path) else None
                    if (
                        master is None
                        or str(master.n_binning) != str(frame.n_binning)
                        or (dtype == CALIB_TYPE_DARK and float(master.exptime) != float(frame.exptime))
                    ):
                        unusable += 1
                        self.logger.warning(
                            f"reuse_recorded_masters: recorded {dtype} {basename} is "
                            f"{'not on disk' if master is None else 'of another binning/exptime'}; "
                            f"{_stem(processed)} takes the normal selection for that role"
                        )
                        continue
                    pins_of.setdefault(_stem(processed), {})[dtype] = (path, master, False)
                    pinned += 1
        self.logger.info(
            f"reuse_recorded_masters: {frames} frame(s) carry a record, {pinned} role(s) pinned to a recorded master, "
            f"{unusable} record(s) unusable"
        )

    def _split_designated_frames(self):
        """Move pinned frames (explicit `images:`; a frame's own record under reuse_recorded_masters) into synthetic groups."""
        self._image_pins = {}
        pins_of = {}
        for dtype, lst in self._designated.items():
            for path, name, _, images in lst:
                for stem in images:
                    if dtype in pins_of.setdefault(stem, {}):
                        raise PreprocessError.ValueError(f"designated_masterframes: two {dtype} pins for {stem}")
                    pins_of[stem][dtype] = (path, name, True)
        if self._reuse_recorded:
            self._add_recorded_pins(pins_of)
        if not pins_of:
            return

        matched = set()
        synthetic = {}
        for i, (raws, masters, sci_dict) in enumerate(self.raw_groups):
            for key in list(sci_dict):
                kept = ([], [])
                for raw, processed in zip(*sci_dict[key]):
                    hit = {s for s in (_stem(raw), _stem(processed)) if s in pins_of}
                    if not hit:
                        kept[0].append(raw)
                        kept[1].append(processed)
                        continue
                    matched |= hit
                    pins = {}
                    for stem in hit:
                        pins.update(pins_of[stem])
                    frame = NameHandler(raw)
                    for dtype, (path, name, explicit) in pins.items():
                        if explicit and (
                            str(name.n_binning) != str(frame.n_binning)
                            or (dtype == CALIB_TYPE_DARK and float(name.exptime) != float(frame.exptime))
                        ):
                            raise PreprocessError.ValueError(
                                f"designated_masterframes: {os.path.basename(path)} does not fit {_stem(raw)} "
                                "(binning, or exptime for a dark)"
                            )
                    pinned = tuple(sorted((dtype, path) for dtype, (path, _, _) in pins.items()))
                    group = synthetic.setdefault((i, pinned), (list(masters), {}))
                    group[1].setdefault(key, ([], []))
                    group[1][key][0].append(raw)
                    group[1][key][1].append(processed)
                if kept[0]:
                    sci_dict[key] = kept
                else:
                    del sci_dict[key]

        if missing := {s for s in set(pins_of) - matched if any(e for _, _, e in pins_of[s].values())}:
            raise PreprocessError.ValueError(f"designated_masterframes: `images:` not in this run: {sorted(missing)}")
        # a natural group left with no science and no raw calibs has nothing left to do
        self.raw_groups = [g for g in self.raw_groups if g[2] or any(g[0])]
        for (_, pinned), (templates, sci_dict) in synthetic.items():
            self._image_pins.update({(len(self.raw_groups), dtype): path for dtype, path in pinned})
            self.raw_groups.append([[[], [], []], templates, sci_dict])

    def _designated_for(self, dtype, template, group_index):
        """The designated master for one group's `dtype` role, or None."""
        if not self._designated.get(dtype) or not template:
            return None
        target = NameHandler(template)
        matches = []
        for path, name, role, images in self._designated[dtype]:
            if images or (role is not None and role != dtype):
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
        if not self._designated and not self._image_pins:
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
                    for path, name, role, images in self._designated[CALIB_TYPE_DARK]
                    if role in (None, "flatdark") and not images and str(name.n_binning) == str(target.n_binning)
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

        self._designation_dispatch.update(self._image_pins)
        for (i, role), path in sorted(self._designation_dispatch.items()):
            if role in self._key_to_index:
                template = self.raw_groups[i][1][self._key_to_index[role]]
                where = f" (in place of {os.path.basename(template)})"
            else:
                where = ""
            self.logger.info(f"[Group {i+1}] Designated {role}: {os.path.basename(path)}{where}")

        unmatched = {p for lst in self._designated.values() for p, _, _, _ in lst} - set(
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
        name, target = NameHandler(designated_file), NameHandler(template)
        if any(str(getattr(name, key)) != str(getattr(target, key)) for key in self._LENIENT_KEYS[dtype]):
            self._ppflag[
                dtype
            ] |= ppflag.PPFLAG_LENIENT_KEYS_IGNORED  # the bit a lenient fetch of the same file carries
        if dtype == CALIB_TYPE_FLAT and name.filter != target.filter:
            # a pin across the filter key: keep the frame out of this filter's default coadd selection
            self._ppflag[dtype] |= ppflag.PPFLAG_HARD_KEYS_IGNORED
            self.logger.warning(
                f"[Group {self._current_group+1}] Designated flat {os.path.basename(designated_file)} is of another "
                f"filter than {os.path.basename(template)}; PPFLAG bit {ppflag.PPFLAG_HARD_KEYS_IGNORED} set"
            )
        self.raw_groups[self._current_group][1][self._key_to_index[dtype]] = designated_file
