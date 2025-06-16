from astropy.table import Table, Column
from pathlib import Path
from datetime import datetime


reference_date = "250430"
reference_state = {"unit1": "31093", "unit2": "31200"}


class CamSwapTracker:
    def __init__(self, changelog_dir, reference_date, reference_state):
        """
        Parameters
        ----------
        changelog_dir : str or Path
            Directory containing 16 changelog files (TSV format).
        reference_date : str
            Date in YYMMDD format when reference_state applies.
        reference_state : dict
            Mapping unit_id -> serial at reference_date.
        """
        self.reference_date = datetime.strptime(reference_date, "%y%m%d")
        self.ref_state = reference_state.copy()
        self.events = self._load_events(changelog_dir)

    def _load_events(self, changelog_dir):
        records = []
        for fpath in Path(changelog_dir).glob("*.tsv"):
            unit = fpath.stem
            # Read TSV into an Astropy Table
            tbl = Table.read(fpath, format="ascii", delimiter="\t")
            # Filter rows where parts == 'cam'
            cam_rows = [row for row in tbl if row["parts"] == "cam"]
            for row in cam_rows:
                # Parse swap comment, e.g. 'swap:31116>31093'
                comment = row["comment"].replace("swap:", "")
                old_serial, new_serial = comment.split(">")
                # Parse date_start field (string or int)
                ds = str(row["date_start"]).zfill(6)
                date = datetime.strptime(ds, "%y%m%d")
                records.append({"unit": unit, "date": date, "old": old_serial, "new": new_serial})
        # Sort records by date
        records.sort(key=lambda x: x["date"])
        return records

    def get_state(self, query_date):
        """
        Return a dict of unit_id -> serial at the given query_date (YYMMDD).
        """
        qd = datetime.strptime(query_date, "%y%m%d")
        state = self.ref_state.copy()
        # Forward or backward apply swaps
        if qd > self.reference_date:
            # apply swaps after reference_date up to qd
            for ev in self.events:
                if self.reference_date < ev["date"] <= qd:
                    state[ev["unit"]] = ev["new"]
        elif qd < self.reference_date:
            # apply swaps before reference_date down to qd in reverse
            for ev in reversed(self.events):
                if qd < ev["date"] <= self.reference_date:
                    state[ev["unit"]] = ev["old"]
        return state
