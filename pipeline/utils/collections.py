from ..const import ALL_GROUP_KEYS


def atleast_1d(x):
    return [x] if not isinstance(x, list) else x


def flatten(nested, max_depth=None):
    """
    Flatten a nested list/tuple up to max_depth levels.

    Args:
        nested (list or tuple): input sequence
        max_depth (int or None): how many levels to flatten.
            None means “flatten completely”.

    Returns:
        list: flattened up to the requested depth
    """

    def _flatten(seq, depth):
        for el in seq:
            if isinstance(el, (list, tuple)) and (max_depth is None or depth < max_depth):
                yield from _flatten(el, depth + 1)
            else:
                yield el

    return list(_flatten(nested, 0))


def most_common_in_dict(counts: dict):
    best = None
    best_count = -1
    for key, cnt in counts.items():
        if cnt > best_count:
            best, best_count = key, cnt
    return best_count, best


def most_common_in_list(seq: list):
    if not seq:
        return None

    counts = {}
    for item in seq:
        counts[item] = counts.get(item, 0) + 1

    return most_common_in_dict(counts)


def equal_in_keys(d1: dict, d2: dict, keys: list):
    # return all(d1[k] == d2[k] for k in keys)
    return all(d1.get(k) == d2.get(k) for k in keys)  # None if key missing


def collapse(seq: list | dict[list], keys=ALL_GROUP_KEYS, raise_error=False, force=False):
    """
    If seq is non-empty and every element equals the first one,
    return the first element; else return seq unchanged.
    """
    if isinstance(seq, list):
        if seq:
            first = seq[0]
        else:
            # raise ValueError("Uncollapsible: input is empty list")
            return seq  # just return empty list
    # dict[str, list]
    elif isinstance(seq, dict):
        return {k: collapse(v) if isinstance(v, list) else v for k, v in seq.items()}
    else:
        # nothing to collapse
        return seq

    # list[dict]
    if isinstance(first, dict):
        common_keys = [k for k in keys if all(k in d for d in seq)]
        if common_keys and all(d[k] == first[k] for d in seq for k in common_keys):
            return first
    # list[str, int, float, ...]
    else:
        if all(x == first for x in seq) or force:
            return first

    if raise_error:
        raise ValueError(f"Uncollapsible: input is not homogeneous: {seq}")
    else:
        return seq
