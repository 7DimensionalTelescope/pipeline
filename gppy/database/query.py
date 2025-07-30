from datetime import date
from typing import List, Dict, Optional, Union, Any, Tuple
import psycopg
import re
from .const import dbname, user, host, port, password, TABLES, ALIASES

# all column names and data types in a table
query_example = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'survey_scienceframe';
    """


# sql injection risk. dev only
def free_query(query: str, params: Optional[List[Any]] = None) -> List[Tuple]:
    """
    Execute a free-form query and return the results.
    """
    conn = psycopg.connect(
        dbname=dbname,
        user=user,
        host=host,
        port=port,
        password=password,
    )
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


###############################################################################


def normalize_types(requested: Optional[List[str]]) -> List[str]:
    """Turn things like ['science','darkframe'] into ['sci','dark']."""
    if not requested:
        return list(TABLES.keys())
    normalized = []
    for t in requested:
        key = t.strip().lower()
        if key in ALIASES:
            normalized.append(ALIASES[key])
        else:
            raise ValueError(f"Unknown image type: {t!r}")
    return sorted(set(normalized))


def get_image_files(
    target_date: Optional[Union[str, date, List[Union[str, date]]]] = None,
    filter_names: Optional[Union[str, List[str]]] = None,
    image_types: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
    target_names: Optional[Union[str, List[str]]] = None,
    # → future params here, e.g. min_exptime, instrument, night_range, etc.
) -> Dict[str, List[Dict]]:
    """
    Core query: any combination of date, filter, types, units, ...
    """

    # normalize date(s)
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    elif isinstance(target_date, list):
        # Convert list of strings to list of dates
        target_date = [date.fromisoformat(d) if isinstance(d, str) else d for d in target_date]

    conn = psycopg.connect(dbname=dbname, user=user, host=host, port=port, password=password)
    results: Dict[str, List[Dict]] = {}

    types = normalize_types(image_types)
    tables = {t: TABLES[t] for t in types}

    with conn.cursor() as cur:
        for img_type, tbl in tables.items():
            # base SELECT + joins
            query = f"""
            SELECT
                sf.id,
                sf.file_path,
                sf.original_filename,
                sf.unified_filename,
                sf.obstime,
                sf.exptime,
                sf.instrument,
                u.name AS unit,
                COALESCE(f.name, NULL) AS filter,
                n.date AS nightdate,
                COALESCE(t.name, NULL) AS target_name
            FROM
                "{tbl}" sf
            JOIN
                facility_unit u ON sf.unit_id = u.id
            JOIN
                survey_night n ON sf.night_id = n.id
            """

            if img_type in ("sci", "flat"):
                query += " LEFT JOIN facility_filter f ON sf.filter_id = f.id"
            else:
                # For bias and dark frames, we don't have filter_id, so we need to handle this differently
                query += " LEFT JOIN facility_filter f ON NULL"

            # Add target join for science frames only
            if img_type == "sci":
                query += " LEFT JOIN survey_target t ON sf.target_id = t.id"
            else:
                # For other frame types, we don't have target_id
                query += " LEFT JOIN survey_target t ON NULL"

            # accumulate WHERE clauses
            clauses: List[str] = []
            params: List[Union[str, date]] = []

            if target_date:
                if isinstance(target_date, list):
                    # Multiple dates
                    placeholders = ",".join(["%s"] * len(target_date))
                    clauses.append(f"n.date IN ({placeholders})")
                    params.extend(target_date)
                else:
                    # Single date
                    clauses.append("n.date = %s")
                    params.append(target_date)

            if filter_names and img_type in ("sci", "flat"):
                if isinstance(filter_names, list):
                    # Multiple filters
                    placeholders = ",".join(["%s"] * len(filter_names))
                    clauses.append(f"f.name IN ({placeholders})")
                    params.extend(filter_names)
                else:
                    # Single filter
                    clauses.append("f.name = %s")
                    params.append(filter_names)
            if units:
                placeholders = ",".join(["%s"] * len(units))
                clauses.append(f"u.name IN ({placeholders})")
                params.extend(units)

            if target_names and img_type == "sci":
                if isinstance(target_names, list):
                    # Multiple targets
                    placeholders = ",".join(["%s"] * len(target_names))
                    clauses.append(f"t.name IN ({placeholders})")
                    params.extend(target_names)
                else:
                    # Single target
                    clauses.append("t.name = %s")
                    params.append(target_names)
            # ... future filters go here ...

            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY sf.obstime;"

            try:
                cur.execute(query, params)
                rows = cur.fetchall()
                results[img_type] = [
                    {
                        "id": row[0],
                        "file_path": row[1],
                        "original_filename": row[2],
                        "unified_filename": row[3],
                        "obstime": row[4],
                        "exptime": row[5],
                        "instrument": row[6],
                        "unit": row[7],
                        "filter": row[8],
                        "nightdate": row[9],
                        "target_name": row[10],
                    }
                    for row in rows
                ]
            except Exception as e:
                print(f"Error querying {tbl}: {e}")
                results[img_type] = []

    conn.close()
    return results


class RawImageQuery:
    """
    Example 1:
    subset = (RawImageQuery()
    .on_date(date.today())
    .for_units(["7DT01"])
    .of_types(["sci"])
    .fetch())

    Example 2:
    subset = RawImageQuery(["2024-10-17"]).fetch()
    """

    def __init__(self, param_list: List[str] = None):
        self._params = {
            "target_date": None,
            "filter_names": None,
            "image_types": None,
            "units": None,
            "target_names": None,
        }
        self._results = None
        if param_list:
            self.classify_parameters(param_list)

    ####################### Fluent Custom Query ###############################

    def on_date(self, d: Union[str, date, List[Union[str, date]]]):
        self._params["target_date"] = d
        return self

    def with_filter(self, filt: Union[str, List[str]]):
        self._params["filter_names"] = filt
        return self

    def for_units(self, units: List[str]):
        self._params["units"] = units
        return self

    def of_types(self, types: List[str]):
        self._params["image_types"] = types
        return self

    def for_target(self, target: Union[str, List[str]]):
        self._params["target_names"] = target
        return self

    # … add more fluent setters …

    def fetch(self) -> Dict[str, List[Dict]]:
        # Filter out None values to avoid passing them to get_image_files
        params = {k: v for k, v in self._params.items() if v is not None}
        self._results = get_image_files(**params)
        return self._results

    ##################### Auto-parsed Query ###################################

    def classify_parameters(self, param_list: List[str]) -> Dict[str, List[str]]:
        """
        Classify parameters from a list into different types.

        Parameters:
        -----------
        param_list : List[str]
            List of parameters to classify

        Returns:
        --------
        Dict[str, List[str]]
            Dictionary with classified parameters:
            - 'dates': List of date strings
            - 'units': List of unit names (7DTXX format)
            - 'filters': List of filter names (u, g, r, i, z, mXXX)
            - 'types': List of image types
        """
        # Initialize lists for each parameter type
        self._params["target_date"] = []
        self._params["units"] = []
        self._params["filter_names"] = []
        self._params["image_types"] = []
        self._params["target_names"] = []

        # Date pattern: YYYY-MM-DD
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        # Unit pattern: 7DT followed by 2 digits (01-20)
        unit_pattern = re.compile(r"^7DT(0[1-9]|1[0-9]|20)$")

        # Filter patterns: single letters or m followed by exactly 3 digits, optionally with 'w' suffix
        filter_pattern = re.compile(r"^[ugriz]$|^m\d{3}w?$")

        for param in param_list:
            param = param.strip()

            # Check if it's a date
            if date_pattern.match(param):
                self._params["target_date"].append(param)
            # Check if it's a unit
            elif unit_pattern.match(param):
                self._params["units"].append(param)
            # Check if it's a filter
            elif filter_pattern.match(param):
                self._params["filter_names"].append(param)
            # Check if it's an image type
            elif param.lower() in ALIASES:
                self._params["image_types"].append(param)
            else:
                # Assume it's a target name (like T02524)
                self._params["target_names"].append(param)

        # Convert single-item lists to single values for compatibility
        if len(self._params["target_date"]) == 1:
            self._params["target_date"] = self._params["target_date"][0]
        elif len(self._params["target_date"]) > 1:
            # Keep as list for multiple dates
            pass

        if len(self._params["filter_names"]) == 1:
            self._params["filter_names"] = self._params["filter_names"][0]
        elif len(self._params["filter_names"]) > 1:
            # Keep as list for multiple filters
            pass

        if len(self._params["target_names"]) == 1:
            self._params["target_names"] = self._params["target_names"][0]
        elif len(self._params["target_names"]) > 1:
            # Keep as list for multiple targets
            pass

        return self._params

    def image_files(self, divide_by_img_type: bool = False) -> Union[Dict[str, List[str]], List[str]]:
        """
        Return full file paths for queried image files.

        Parameters:
        -----------
        divide_by_img_type : bool, default True
            If True, return paths organized by image type.
            If False, return a flat list of all paths.

        Returns:
        --------
        Union[Dict[str, List[str]], List[str]]
            If divide_by_img_type=True: Dictionary with image type as key and list of full file paths as value
            If divide_by_img_type=False: List of all file paths
        """
        if self._results is None:
            self.fetch()

        if divide_by_img_type:
            return {
                img_type: [file_info["file_path"] for file_info in files] for img_type, files in self._results.items()
            }
        else:
            # Flatten all file paths from all image types
            all_paths = []
            for files in self._results.values():
                all_paths.extend([file_info["file_path"] for file_info in files])
            return all_paths
