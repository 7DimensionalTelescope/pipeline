"""
Human-verdict cascade of a master frame: PPFLAG bit 4 (a master that failed SANITY was used).

A verdict on a master recomputes bit 4 on what consumed it: derived masters first, then
singles. Writes stop at singles; coadds, whites, diffs and configs below them are reported,
never rewritten. Only bit 4 is recomputed; bits 1/8/16/32/64 are fetch-time decisions.
Undo is `revoke_master`: the verdict withdrawn, then the same recompute.

    from pipeline.services.inspection import cascade_master
    cascade_master("/lyman/data2/master_frame/2023-11-30/7DT06/bias_7DT06_20231201_1x1_gain1_C36666.fits",
                   dry_run=True)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Union

from astropy.io import fits

from ..const import CALIB_TYPES, IMAGE_TYPE_SINGLE
from ..utils import atleast_1d


class Change(NamedTuple):
    """Planned PPFLAG: path, old card (None when absent), new value."""

    path: str
    old: Optional[int]
    new: int


@dataclass
class Closure:
    """Downstream of masters: derived masters, singles, products by type, configs."""

    derived_masters: List[tuple] = field(default_factory=list)
    singles: List[tuple] = field(default_factory=list)
    products: Dict[str, List[tuple]] = field(default_factory=dict)
    configs: List[tuple] = field(default_factory=list)


@dataclass
class CascadeReport:
    """Cascade outcome: closure counts by type, configs, planned and applied changes, seconds."""

    master: str
    counts: Dict[str, int]
    configs: List[tuple]
    planned: List[Change]
    applied: Optional[dict]
    elapsed: float


def master_closure(masters: Union[str, List[str]], max_depth: int = 10) -> Closure:
    """Downstream closure of masters: derived masters, singles, products by type, configs; read-only."""
    from .database.image_qa_dependency import ImageQADependency, _qa_rows

    dependency = ImageQADependency()
    seeds = {row[0] for rows in _qa_rows(dependency.execute_query, atleast_1d(masters)).values() for row in rows}
    closure = Closure()
    if not seeds:
        return closure
    impacted = [row for row in dependency.impacted_images(sorted(seeds), max_depth) if row[0] not in seeds]
    paths = dict(
        dependency.execute_query(
            "SELECT id, image_path FROM image_qa WHERE id = ANY(%s)", ([row[0] for row in impacted],)
        )
    )
    seen = set()
    for image_id, _, image_type, image_name in impacted:
        path = paths.get(image_id)
        if (path or image_name) in seen:
            continue
        seen.add(path or image_name)
        if image_type in CALIB_TYPES:
            closure.derived_masters.append((image_id, path, image_type))
        elif image_type == IMAGE_TYPE_SINGLE:
            closure.singles.append((image_id, path))
        else:
            closure.products.setdefault(image_type, []).append((image_id, image_name, path))
    closure.configs = dependency.impacted_configs(sorted(seeds), max_depth)
    return closure


def _source_edges(images: List[str]) -> Dict[str, List[tuple]]:
    """image -> [(role, source path)] of its bias/dark/flat sources, from image_qa_dependency."""
    from .database.image_qa_dependency import ImageQADependency, _qa_rows

    dependency = ImageQADependency()
    images_of = {}
    for image, rows in _qa_rows(dependency.execute_query, images).items():
        for row in rows:
            images_of.setdefault(row[0], []).append(image)
    if not images_of:
        return {}
    edges = dependency.execute_query(
        "SELECT d.derived_image_id, d.dependency_role, s.image_path FROM image_qa_dependency d"
        " JOIN image_qa s ON s.id = d.source_image_id"
        " WHERE d.derived_image_id = ANY(%s) AND d.dependency_role = ANY(%s)",
        (list(images_of), list(CALIB_TYPES)),
    )
    sources = {}
    for derived_id, role, source_path in edges:
        for image in images_of[derived_id]:
            if source_path and (role, source_path) not in sources.setdefault(image, []):
                sources[image].append((role, source_path))
    return sources


def _sources_from_db(images: List[str]) -> Dict[str, List[str]]:
    """image -> paths of its bias/dark/flat sources, from image_qa_dependency."""
    return {image: list(dict.fromkeys(path for _, path in edges)) for image, edges in _source_edges(images).items()}


def _sanity_from_db(images: List[str]) -> Dict[str, Optional[bool]]:
    """image -> image_qa.sanity; False when any of its rows is False."""
    from .database.image_qa_dependency import ImageQADependency, _qa_rows

    rows = _qa_rows(ImageQADependency().execute_query, images)
    return {
        image: False if any(row[3] is False for row in found) else found[0][3] for image, found in rows.items() if found
    }


def _sanity_from_headers(images: List[str]) -> Dict[str, Optional[bool]]:
    """image -> SANITY card of its FITS header."""
    return {image: fits.getheader(image).get("SANITY") for image in images if os.path.exists(image)}


def _ppflag_card(image: str) -> Optional[int]:
    """PPFLAG card of the FITS header; None when absent or no file."""
    if not os.path.exists(image):
        return None
    value = fits.getheader(image).get("PPFLAG")
    return None if value is None else int(value)


def sanity_f_bit_plan(
    images: Union[str, List[str]],
    sources_of: Optional[Callable[[List[str]], Dict[str, List[str]]]] = None,
    sanity_of: Optional[Callable[[List[str]], Dict[str, Optional[bool]]]] = None,
) -> List[Change]:
    """PPFLAG bit-4 changes: a used master, at any ingredient depth, with SANITY False; changed images only."""
    sanity_f_used = 4  # preprocess.ppflag.PPFLAG_SANITY_F_USED; importing preprocess imports services.queue (SIGINT)
    images = list(dict.fromkeys(image for image in atleast_1d(images) if image))
    sources_of = sources_of or _sources_from_db
    sanity_of = sanity_of or _sanity_from_db
    direct = sources_of(images)
    used = {image: set(direct.get(image, [])) for image in images}
    frontier, seen = sorted(set().union(*used.values())), set()
    while frontier:  # a flat's flatdark's bias sits three levels down
        seen.update(frontier)
        deeper = sources_of(frontier)
        for masters in used.values():
            masters.update(*(deeper.get(source, []) for source in list(masters)))
        frontier = sorted({source for sources in deeper.values() for source in sources} - seen)
    sanity = sanity_of(sorted(set().union(*used.values())))
    with ThreadPoolExecutor(8) as pool:
        olds = list(pool.map(_ppflag_card, images))

    changes = []
    for image, old in zip(images, olds):
        any_f = any(sanity.get(master) is False for master in used[image])
        if old is None:
            if any_f:
                changes.append(Change(image, None, sanity_f_used))
            continue
        new = (old & ~sanity_f_used) | (sanity_f_used if any_f else 0)
        if new != old:
            changes.append(Change(image, old, new))
    return changes


def _write_ppflag(change: Change) -> str:
    """PPFLAG card on the FITS, then its sidecar; the sidecar's outcome."""
    from ..utils.header import update_padded_header_file, update_padded_header_smart

    cards = {"PPFLAG": change.new}
    update_padded_header_smart(change.path, cards)
    return update_padded_header_file(change.path, cards)


def apply_ppflag_changes(changes: List[Change], use_database: bool = True, workers: int = 8) -> dict:
    """Write planned PPFLAGs: FITS card and sidecar in a thread pool, then image_qa.ppflag on the main thread."""
    report = {
        "headers_written": [],
        "sidecars_written": [],
        "sidecars_unchanged": [],
        "sidecars_missing": [],
        "files_missing": [],
        "image_qa_updated": [],
        "image_qa_missing": [],
    }
    present = []
    for change in changes:
        if os.path.exists(change.path):
            present.append(change)
        else:
            report["files_missing"].append(change.path)

    with ThreadPoolExecutor(workers) as pool:
        for change, sidecar in zip(present, pool.map(_write_ppflag, present)):
            report["headers_written"].append(change.path)
            report[f"sidecars_{sidecar}"].append(change.path)

    if use_database and present:
        from .database.image_qa import ImageQA
        from .database.image_qa_dependency import _qa_rows

        image_qa = ImageQA()
        rows = _qa_rows(image_qa.execute_query, [change.path for change in present])
        for change in present:
            if not rows.get(change.path):
                report["image_qa_missing"].append(change.path)
            for row in rows.get(change.path, []):
                image_qa.update_data(row[0], ppflag=change.new)
                report["image_qa_updated"].append(row[0])
    return report


def cascade_master(
    master: str, dry_run: bool = False, use_database: bool = True, sanity: Optional[bool] = None
) -> CascadeReport:
    """Recompute PPFLAG bit 4 below a master: derived masters, then singles; the rest is counted; `sanity` assumes the master's verdict (dry run before writing it)."""
    start = time.perf_counter()
    closure = master_closure(master)
    counts = {}
    for _, _, image_type in closure.derived_masters:
        counts[image_type] = counts.get(image_type, 0) + 1
    counts[IMAGE_TYPE_SINGLE] = len(closure.singles)
    counts.update({image_type: len(rows) for image_type, rows in closure.products.items()})

    images = [row[1] for row in closure.derived_masters] + [row[1] for row in closure.singles]
    # without the database the verdict lives only in the headers, so SANITY is read there too
    read = _sanity_from_db if use_database else _sanity_from_headers
    name = os.path.basename(master)

    def sanity_of(paths: List[str]) -> Dict[str, Optional[bool]]:
        found = read(paths)
        return found if sanity is None else {**found, **{p: sanity for p in paths if os.path.basename(p) == name}}

    planned = sanity_f_bit_plan(images, sanity_of=sanity_of)
    applied = None if dry_run else apply_ppflag_changes(planned, use_database=use_database)
    return CascadeReport(master, counts, closure.configs, planned, applied, time.perf_counter() - start)


def revoke_master(master: str, use_database: bool = True) -> dict:
    """Withdraw a master's verdict (SANITY from its own QA criteria), then recompute the cascade."""
    from ..path.name import NameHandler
    from .checker import Checker
    from .inspection import revoke_inspection

    sanity = Checker().apply_qa_criteria(header=fits.getheader(master), dtype=NameHandler(master).type.image_type)
    revoked = revoke_inspection(master, use_database=use_database, sanity=sanity)
    return {"revoke": revoked, "cascade": cascade_master(master, use_database=use_database)}
