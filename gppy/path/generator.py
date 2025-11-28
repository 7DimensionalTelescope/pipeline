import os
from glob import glob
from typing import Iterator

from gppy.const import PROCESSED_DIR
from gppy.path.path import NameHandler


def iter_single_images(pattern: str, cross_date: bool = True):
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

    except IndexError:
        date_pattern = pattern
        return iter_processed(date_pattern=date_pattern, type_pattern="singles")

    except Exception as e:
        raise ValueError(f"Invalid pattern: {pattern}\n{e}")


def iter_configs(date_pattern: str):
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

    return iter_processed(date_pattern=date_pattern, type_pattern="", filename_pattern="*.yml")


def iter_stacked_images(date_pattern: str):
    """date_pattern: e.g. "2025-11*", "2025-*", etc."""
    return iter_processed(date_pattern=date_pattern, type_pattern="stacked")


def iter_catalogs(date_pattern: str, type_pattern: str = "singles"):
    return iter_processed(date_pattern=date_pattern, type_pattern=type_pattern, filename_pattern="*_cat.fits")


def iter_processed(
    date_pattern: str = "2025-11*",  # e.g. "2025-10-*", "2025-*", etc.
    obj_pattern: str = "*",
    filter_pattern: str = "*",
    type_pattern: str = "singles",
    filename_pattern: str = "*s.fits",  # pattern for files in singles/
    base: str = PROCESSED_DIR,
) -> Iterator[str]:
    """
    Lazily yield FITS files matching the pattern:
        <base>/<date_pattern>/<obj_pattern>/<filter_pattern>/<type_pattern>/<filename_pattern>

    This walks the tree level by level with smaller globs instead of
    doing one huge glob over everything.
    """

    # 1) dates: /.../2025-1*
    for date_dir in sorted(glob(os.path.join(base, date_pattern))):
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
