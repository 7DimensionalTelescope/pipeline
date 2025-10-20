from functools import reduce


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
