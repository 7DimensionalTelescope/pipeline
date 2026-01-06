import os
from typing import Optional, Tuple, Union


def set_umask(mask: Union[int, str]="0002") -> Optional[Tuple[int, int]]:
    """
    Set process umask. Accepts:
      - int like 0o022 / 0o002
      - str like "0022" / "0002" (interpreted as octal)

    Prints only if changed.
    Returns (old_mask, new_mask) if changed, else None.
    """
    if isinstance(mask, str):
        s = mask.strip()
        if s.startswith(("0o", "0O")):
            new = int(s, 8)
        else:
            # treat as octal digits like shell output: "0022"
            new = int(s, 8)
    elif isinstance(mask, int):
        new = mask
    else:
        raise TypeError("mask must be int or str")

    if not (0 <= new <= 0o777):
        raise ValueError(f"umask out of range: {new:o}")

    old = os.umask(new)  # sets to new, returns previous
    if old != new:
        print(f"umask changed: {old:04o} -> {new:04o}")
        return old, new
    return None