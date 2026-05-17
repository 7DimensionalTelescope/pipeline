import os
import re
from typing import Literal
import numpy as np
from astropy.table import Table
from datetime import date, datetime, timedelta

from .environ import REF_DIR

WALL_KINDS = ("bias", "dark", "flat")


def get_changelog(unit: int) -> Table:
    changelog = os.path.join(REF_DIR, f"InstrumEvent/changelog_unit{unit}.txt")
    # tbl = Table.read(changelog, format="ascii.tab", guess=False, fast_reader=False)
    tbl = Table.read(changelog, format="ascii.basic", delimiter="\t", guess=False, fast_reader=False)

    tbl["nightdate"] = [
        log_date_to_nightdate(d) if np.ma.is_masked(nd) else datetime.strptime(str(nd), "%Y-%m-%d").date()
        for d, nd in zip(tbl["date"], tbl["nightdate"])
    ]

    return tbl


def log_date_to_nightdate(log_date: str) -> date:
    """
    Caution: the logs are in KST, telescopes are in CLST.
    This function returns in CLST.
    Assumes logging (KST) happened half a day later than the actual event (CLST).

    InstrumEvent log has ~1 day uncertainty, as the logging sometimes happen
    real-time, sometimes after the event.

    Assuming half-day delay, an event happens on nightdate = log date - 1 day.
    The result of the instrumental event usually takes effect the next day.
    """
    # KST
    date_kst = datetime.strptime(str(log_date), "%y%m%d").date()
    # CLST
    # a compromised mean offset between KST date and CLST nightdate, given +-1 day uncertainty in the log.
    nightdate = date_kst - timedelta(days=0)  # =1
    return nightdate


def get_cam_events(unit: int, swap_only: bool = False):
    """
    Caution: the logs are in KST, telescopes are in CLST.
    This function returns in CLST.

    Assumes logging (KST) happened half a day later than the actual event (CLST).
    """
    tbl = get_changelog(unit)

    is_cam = tbl["parts"] == "cam"
    # is_install = np.char.startswith(comments, "install:")
    # is_uninstall = np.char.startswith(comments, "uninstall:")
    # is_swap = np.char.startswith(comments, "swap:")

    # combine all conditions with bitwise ops
    mask = is_cam  # & (is_install | is_uninstall | is_swap)
    filtered_tbl = tbl[mask]
    cam_event_tbl = Table()
    cam_event_tbl["nightdate"] = [log_date_to_nightdate(s) for s in filtered_tbl["date"]]
    # camswap_tbl["serial"] = [s.split(":")[-1].split(">")[-1] for s in filtered_tbl["comment"]]
    cam_event_tbl["serial"] = filtered_tbl["comment"]
    if swap_only:
        _ = [bool(re.match(r"\d+", str(s))) for s in cam_event_tbl["serial"]]
        cam_event_tbl = cam_event_tbl[cam_event_tbl["serial"].astype(str) != ""]
        return cam_event_tbl[np.asarray(_)]

    return cam_event_tbl


def normalize_wall(
    t: Table,
    date_col: str = "nightdate",
    wall_col: str = "wall",
    *,
    dtype: Literal["bias", "dark", "flat"],
) -> Table:
    out = Table()
    dates = np.unique(t[date_col])
    out[date_col] = dates
    out[f"wall_{dtype}"] = np.zeros(len(dates), dtype=bool)

    for i, d in enumerate(dates):
        for wall in t[t[date_col] == d][wall_col]:
            if np.ma.is_masked(wall):
                continue
            vals = {x.strip() for x in str(wall).split(",") if x.strip()}
            if dtype in vals:
                out[f"wall_{dtype}"][i] = True
                break

    return out


def get_masterframe_walls(
    nightdate: str | date,
    unit: int,
    *,
    dtype: Literal["bias", "dark", "flat"],
) -> tuple[date | None, date | None]:
    nightdate = datetime.strptime(nightdate, "%Y-%m-%d").date()

    changelog = get_changelog(unit)
    tbl = normalize_wall(changelog, dtype=dtype)
    tbl.sort("nightdate")

    nightdates = np.array(tbl["nightdate"], dtype=object)

    idx = np.searchsorted(nightdates, nightdate, side="right")
    lo = idx - 1
    hi = idx

    lower = nightdates[lo] if lo >= 0 else None  # inclusive
    upper = nightdates[hi] - timedelta(days=1) if hi < len(nightdates) else None  # exclusive
    return lower, upper
