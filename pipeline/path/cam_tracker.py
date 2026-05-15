import json
from datetime import datetime

from ..const.environ import INSTRUM_STATUS_DICT
from ..const.instrum_log import get_cam_events


def get_current_camera_serial(unit: int) -> str:
    ref_path = INSTRUM_STATUS_DICT  # independently updated reference file
    with open(ref_path) as f:
        data = json.load(f)
    current_serial: str = data[f"7DT{unit:02d}"]["Camera"]["name"]
    return current_serial


def get_camera_serial(unit: int, query_date: str):
    """
    query_date MUST BE nightdate
    WARNING: assumes C3 if serial unavailable
    """
    # 1) load camswap history
    cam_swap_tbl = get_cam_events(unit, swap_only=True)
    dates = list(cam_swap_tbl["nightdate"])
    serials = [str(s) for s in cam_swap_tbl["serial"]]
    # serials = list(swaps_tbl["serial"])

    # 2) parse query_date ("YYYY-MM-DD" or "YYYYMMDD")
    fmt = "%Y-%m-%d"  # if "-" in query_date else "%Y%m%d"
    qdate = datetime.strptime(query_date, fmt).date()

    # 3) if qdate <= last nightdate, pick the most recent past event
    last_date = dates[-1]
    if qdate <= last_date:
        # find all indices where date <= qdate
        idxs = [i for i, d in enumerate(dates) if d <= qdate]
        if not idxs:
            # raise ValueError(f"No camera serial known on or before {query_date}")
            fallback_serial = f"3{unit:04}"
            print(f"No camera serial known <= {query_date} for unit{unit}. Using {fallback_serial}")
            # return f"3{str(str(unit)*4)[:4]}"
            return fallback_serial
        last_idx = max(idxs)
        return serials[last_idx]

    # 4) else (qdate > last_date): compare with current reference
    current_serial = get_current_camera_serial(unit)

    # if it matches the last-swapped serial, return it, otherwise error
    if (current_serial == serials[-1]) or current_serial == "":
        return serials[-1]
    else:
        raise ValueError(
            f"Current serial ({current_serial}) of unit{unit} does not match the last "
            f"reported serial ({serials[-1]}) on {last_date.date()}\n"
            f"Update ref/InstrumEvent/changelog_unit{int(unit)}.txt or check {INSTRUM_STATUS_DICT}"
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
