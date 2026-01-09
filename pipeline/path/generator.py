import os
from glob import glob
from typing import Iterator

from ..const import PROCESSED_DIR, MASTER_FRAME_DIR
from .name import NameHandler


def iter_single_images(pattern: str, cross_date: bool = True, reverse: bool = False):
    """
    pattern is either "2025-11*", "2025-*", ...,
    or a full filename like /lyman/data2/processed/2025-11-20/T15645/m825/singles/T15645_m825_7DT07_20251121_052846_100s.fits
    """

    # Assume it's date_pattern if NameHandler can't parse it
    try:
        name = NameHandler(pattern)
        return iter_processed(
            date_pattern=name.date if not cross_date else "*",
            obj_pattern=name.obj,
            filter_pattern=name.filter,
            type_pattern="singles",
        )

    except (UnboundLocalError, IndexError):
        date_pattern = pattern
        return iter_processed(date_pattern=date_pattern, type_pattern="singles", reverse=reverse)

    except Exception as e:
        raise ValueError(f"Invalid pattern: {pattern}\n{e}")


def iter_config(date_pattern: str, reverse: bool = False):
    """
    pattern is "2025-11*", "2025-*", ...,
    """

    # no namehandler support for configs yet
    # if os.path.isfile(pattern) and pattern.endswith(".yml"):
    #     os.path.dirname(pattern)

    #     return iter_processed(
    #         date_pattern=name.date if not cross_date else "*",
    #         obj_pattern=name.obj,
    #         filter_pattern=name.filter,
    #         type_pattern="singles",
    #     )

    return iter_processed(date_pattern=date_pattern, type_pattern="", filename_pattern="*.yml", reverse=reverse)


def iter_coadd_images(date_pattern: str, reverse: bool = False):
    """date_pattern: e.g. "2025-11*", "2025-*", etc."""
    return iter_processed(date_pattern=date_pattern, type_pattern="coadd", reverse=reverse)


def iter_catalog(date_pattern: str, type_pattern: str = "singles", reverse: bool = False):
    return iter_processed(
        date_pattern=date_pattern, type_pattern=type_pattern, filename_pattern="*_cat.fits", reverse=reverse
    )


def iter_processed(
    date_pattern: str = "2025-11*",  # e.g. "2025-10-*", "2025-*", etc.
    obj_pattern: str = "*",
    filter_pattern: str = "*",
    type_pattern: str = "singles",
    filename_pattern: str = "*s.fits",  # pattern for files in singles/
    base: str = PROCESSED_DIR,
    reverse: bool = False,
) -> Iterator[str]:
    """
    Lazily yield FITS files matching the pattern:
        <base>/<date_pattern>/<obj_pattern>/<filter_pattern>/<type_pattern>/<filename_pattern>

    This walks the tree level by level with smaller globs instead of
    doing one huge glob over everything.

    Args:
        reverse: If True, iterate in reverse order (default: False)
    """

    # 1) dates: /.../2025-1*
    for date_dir in sorted(glob(os.path.join(base, date_pattern)), reverse=reverse):
        if not os.path.isdir(date_dir):
            continue

        # 2) targets: /.../<date>/<obj_pattern>
        for target_dir in sorted(glob(os.path.join(date_dir, obj_pattern))):
            if not os.path.isdir(target_dir):
                continue

            # 3) filters: /.../<date>/<target>/<filter_pattern>
            for filter_dir in sorted(glob(os.path.join(target_dir, filter_pattern))):
                if not os.path.isdir(filter_dir):
                    continue

                # If type_pattern is "", stay in filter_dir
                search_dir = os.path.join(filter_dir, type_pattern) if type_pattern else filter_dir
                if not os.path.isdir(search_dir):
                    continue

                # 4) files
                for path in sorted(glob(os.path.join(search_dir, filename_pattern))):
                    yield path


def iter_masterframe(
    date_pattern: str = "*",  # e.g. "2025-12-11", "2025-12-*", "2025-*"
    unit_pattern: str = "*",  # e.g. "7DT03"
    type_pattern: str = "*",  # e.g. "biassig", "dark_100s", "flat_g", "bpmask_100s"
    base: str = MASTER_FRAME_DIR,
) -> Iterator[str]:
    """
    Lazily yield master frame FITS files matching:
        <base>/<date_pattern>/<unit_pattern>/<type_pattern*.fits>

    Directory layout is assumed to be consistent with:
        /lyman/data2/master_frame/2025-12-11/7DT03/<files>

    `type_pattern` can be either:
      - A prefix like "biassig", "dark_100s", "flat_g" (we append "*.fits")
      - A glob like "flat_*_gain2750*.fits" (used as-is)
      - A filename ending with ".fits" (used as-is)
    """

    def _to_filename_glob(p: str) -> str:
        # If user already provided a glob or explicit filename, use it as-is.
        if any(ch in p for ch in ["*", "?", "["]) or p.endswith(".fits"):
            return p
        # Otherwise treat it as a prefix.
        return f"{p}*.fits"

    filename_glob = _to_filename_glob(type_pattern)

    # 1) dates: /.../2025-12-*
    for date_dir in sorted(glob(os.path.join(base, date_pattern))):
        if not os.path.isdir(date_dir):
            continue

        # 2) units: /.../<date>/7DT03
        for unit_dir in sorted(glob(os.path.join(date_dir, unit_pattern))):
            if not os.path.isdir(unit_dir):
                continue

            # 3) files
            for path in sorted(glob(os.path.join(unit_dir, filename_glob))):
                yield path
