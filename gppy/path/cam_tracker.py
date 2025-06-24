import os
import re
import json
import numpy as np
from astropy.table import Table, Column, vstack
from pathlib import Path
from datetime import datetime, timedelta

from ..const import REF_DIR, INSTRUM_STATUS_DICT


def get_cam_events(unit: int):
    """Caution: the logs are in KST, telescopes are in CLST."""
    changelog = os.path.join(REF_DIR, f"InstrumEvent/changelog_unit{unit}.txt")
    tbl = Table.read(changelog, format="ascii")

    is_cam = tbl["parts"] == "cam"
    # is_install = np.char.startswith(comments, "install:")
    # is_uninstall = np.char.startswith(comments, "uninstall:")
    # is_swap = np.char.startswith(comments, "swap:")

    # combine all conditions with bitwise ops
    mask = is_cam  # & (is_install | is_uninstall | is_swap)
    filtered_tbl = tbl[mask]
    camswap_tbl = Table()
    # KST
    date_kst = [datetime.strptime(str(s), "%y%m%d") for s in filtered_tbl["date"]]
    # CLST
    camswap_tbl["nightdate"] = [s - timedelta(days=1) for s in date_kst]
    # camswap_tbl["serial"] = [s.split(":")[-1].split(">")[-1] for s in filtered_tbl["comment"]]
    camswap_tbl["serial"] = filtered_tbl["comment"]

    return camswap_tbl


def get_current_camera_serial(unit: int) -> str:
    ref_path = INSTRUM_STATUS_DICT  # independently updated reference file
    with open(ref_path) as f:
        data = json.load(f)
    current_serial: str = data[f"7DT{unit:02d}"]["Camera"]["name"]
    return current_serial


def get_camera_serial(unit: int, query_date: str):
    # 1) load camswap history
    swaps_tbl = get_cam_events(unit)
    _ = [bool(re.match(r"\d+", str(s))) for s in swaps_tbl["serial"]]
    swaps_tbl = swaps_tbl[np.asarray(_)]
    dates = list(swaps_tbl["nightdate"])
    serials = [str(s) for s in swaps_tbl["serial"]]
    # serials = list(swaps_tbl["serial"])

    # 2) parse query_date ("YYYY-MM-DD" or "YYYYMMDD")
    fmt = "%Y-%m-%d" if "-" in query_date else "%Y%m%d"
    qdate = datetime.strptime(query_date, fmt)

    # 3) if qdate <= last nightdate, pick the most recent past event
    last_date = dates[-1]
    if qdate <= last_date:
        # find all indices where date <= qdate
        idxs = [i for i, d in enumerate(dates) if d <= qdate]
        if not idxs:
            raise ValueError(f"No camera serial known on or before {query_date}")
        last_idx = max(idxs)
        return serials[last_idx]

    # 4) else (qdate > last_date): compare with current reference
    current_serial = get_current_camera_serial(unit)

    # if it matches the last-swapped serial, return it, otherwise error
    if (current_serial == serials[-1]) or current_serial == "":
        return serials[-1]
    else:
        raise ValueError(
            f"Reference serial ({current_serial}) does not match last known "
            f"swapped serial ({serials[-1]}) as of {last_date.date()}"
        )


# reference_date = "250430"
# reference_state = {"unit1": "31093", "unit2": "31200"}


# class CamSwapTracker:
#     def __init__(self, changelog_dir, reference_date, reference_state):
#         """
#         Parameters
#         ----------
#         changelog_dir : str or Path
#             Directory containing 16 changelog files (TSV format).
#         reference_date : str
#             Date in YYMMDD format when reference_state applies.
#         reference_state : dict
#             Mapping unit_id -> serial at reference_date.
#         """
#         self.reference_date = datetime.strptime(reference_date, "%y%m%d")
#         self.ref_state = reference_state.copy()
#         self.events = self._load_events(changelog_dir)

#     def _load_events(self, changelog_dir):
#         records = []
#         for fpath in Path(changelog_dir).glob("*.tsv"):
#             unit = fpath.stem
#             # Read TSV into an Astropy Table
#             tbl = Table.read(fpath, format="ascii", delimiter="\t")
#             # Filter rows where parts == 'cam'
#             cam_rows = [row for row in tbl if row["parts"] == "cam"]
#             for row in cam_rows:
#                 # Parse swap comment, e.g. 'swap:31116>31093'
#                 comment = row["comment"].replace("swap:", "")
#                 old_serial, new_serial = comment.split(">")
#                 # Parse date_start field (string or int)
#                 ds = str(row["date_start"]).zfill(6)
#                 date = datetime.strptime(ds, "%y%m%d")
#                 records.append({"unit": unit, "date": date, "old": old_serial, "new": new_serial})
#         # Sort records by date
#         records.sort(key=lambda x: x["date"])
#         return records

#     def get_state(self, query_date):
#         """
#         Return a dict of unit_id -> serial at the given query_date (YYMMDD).
#         """
#         qd = datetime.strptime(query_date, "%y%m%d")
#         state = self.ref_state.copy()
#         # Forward or backward apply swaps
#         if qd > self.reference_date:
#             # apply swaps after reference_date up to qd
#             for ev in self.events:
#                 if self.reference_date < ev["date"] <= qd:
#                     state[ev["unit"]] = ev["new"]
#         elif qd < self.reference_date:
#             # apply swaps before reference_date down to qd in reverse
#             for ev in reversed(self.events):
#                 if qd < ev["date"] <= self.reference_date:
#                     state[ev["unit"]] = ev["old"]
#         return state
