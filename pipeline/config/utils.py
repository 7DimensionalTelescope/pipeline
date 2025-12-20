from functools import reduce
import os
from ..const import PROCESSED_DIR, TOO_DIR


def find_config(config: str, is_too: bool = False, return_class=False) -> dict:
    import re

    config = config.replace(".yml", "")
    BASE_DIR = TOO_DIR if is_too else PROCESSED_DIR

    # Find date pattern (YYYY-MM-DD)
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    date_match = re.search(date_pattern, config)

    if date_match:
        # Date found - extract date and parts before/after
        date = date_match.group()
        date_start = date_match.start()
        date_end = date_match.end()

        # Part before date
        before_date = config[:date_start].rstrip("_")
        # Part after date (if any)
        after_date = config[date_end:].lstrip("_")

        if before_date:
            # Format: {obj}_{filt}_{date} or {obj}_{filt}_{date}_{unit}
            # Split by underscore, last part before date is filter, rest is object
            before_parts = before_date.split("_")
            if len(before_parts) >= 2:
                # Has object and filter
                filt = before_parts[-1]  # Last part is filter
                obj = "_".join(before_parts[:-1])  # Everything before filter is object
                full_path = f"{BASE_DIR}/{date}/{obj}/{filt}/{config}.yml"
                if return_class:
                    from .sciprocess import SciProcConfiguration

                    return SciProcConfiguration.from_config(full_path, is_too=is_too)
            else:
                # Only one part before date - treat as filter, no object
                filt = before_parts[0]
                full_path = f"{BASE_DIR}/{date}/{filt}/{config}.yml"
                if return_class:
                    from .sciprocess import SciProcConfiguration

                    return SciProcConfiguration.from_config(full_path, is_too=is_too)
        else:
            # Format: {date}_{unit} - date is first
            if after_date:
                unit = after_date
                full_path = f"{BASE_DIR}/{date}/{config}.yml"
            else:
                # Just date
                full_path = f"{BASE_DIR}/{date}/{config}.yml"

            if return_class:
                from .preprocess import PreprocConfiguration

                return PreprocConfiguration.from_config(full_path, is_too=is_too)
    else:
        # No date pattern found - fall back to old logic
        args = config.split("_")
        if len(args) >= 3:
            obj, filt, date = args[:3]
            full_path = f"{BASE_DIR}/{date}/{obj}/{filt}/{config}.yml"
            if return_class:
                from .sciprocess import SciProcConfiguration

                return SciProcConfiguration.from_config(full_path, is_too=is_too)
        elif len(args) == 2:
            date, unit = args[:2]
            full_path = f"{BASE_DIR}/{date}/{config}.yml"
            if return_class:
                from .preprocess import PreprocConfiguration

                return PreprocConfiguration.from_config(full_path, is_too=is_too)
        else:
            raise ValueError(f"Invalid config format: {config}")

    if os.path.exists(full_path):
        return full_path
    else:
        raise FileNotFoundError(f"Config file not found: {full_path}")


def get_filter_from_config(config: str) -> str:
    import re

    config = os.path.basename(config).replace(".yml", "")

    # Find date pattern (YYYY-MM-DD)
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    date_match = re.search(date_pattern, config)

    if date_match:
        # Date found - extract filter (last part before date)
        date_start = date_match.start()
        before_date = config[:date_start].rstrip("_")

        if before_date:
            before_parts = before_date.split("_")
            if len(before_parts) >= 2:
                # Last part before date is filter
                return before_parts[-1]
            elif len(before_parts) == 1:
                # Only one part - treat as filter
                return before_parts[0]

    # Fall back to old logic
    args = config.split("_")
    if len(args) >= 3:
        return args[1]  # Second part is filter in old format

    return None


def merge_dicts(base: dict, updates: dict) -> dict:
    """
    Recursively merge the updates dictionary into the base dictionary.

    For each key in the updates dictionary:
    - If the key exists in base and both values are dictionaries,
        then merge these dictionaries recursively.
    - Otherwise, set or override the value in base with the one from updates.

    Args:
        base (dict): The original base configuration dictionary.
        updates (dict): The new configuration dictionary with updated values.

    Returns:
        dict: The merged dictionary containing updates from the new configuration.
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def get_key(config, key, default=None):
    """
    Safely walk through attributes in `key` (e.g. "photometry.path.ref_ris_dir").
    Returns the final value or `default` if any step is missing or None.
    """

    def _get(o, attr):
        return getattr(o, attr, None) if o is not None else None

    result = reduce(_get, key.split("."), config)
    return result if result is not None else default


def merge_missing(dst, src, exclude_top_level=None, _path_prefix=""):
    """
    Recursively copy only missing keys from src -> dst.
    - If src[k] is a dict, recurse.
    - If src[k] is scalar, set it only if k not in dst or dst[k] is None.
    Returns a list of dotted key paths that were added.

    `In-place` operation. dst is modified in place.

    Top-level keys in exclude_top_level are skipped. If exclude_top_level is None, all top-level keys are included.

    _path_prefix is an internal variable to track the path of the current key being processed.
    User should not set this variable.
    """
    if exclude_top_level is None:
        exclude_top_level = set()

    added = []

    if not isinstance(src, dict):
        return added

    for k, v in src.items():
        # honor top-level exclusions
        if not _path_prefix and k in exclude_top_level:
            continue

        new_path = f"{_path_prefix}.{k}" if _path_prefix else k

        if isinstance(v, dict):
            if k not in dst or not isinstance(dst.get(k), dict):
                dst.setdefault(k, {})
                added.append(new_path)
            added.extend(merge_missing(dst[k], v, exclude_top_level, new_path))
        else:
            if k not in dst or dst.get(k) is None:
                dst[k] = v
                added.append(new_path)

    return added
