from datetime import date
from typing import List, Dict, Optional, Union, Any, Tuple
import psycopg
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
    target_date: Optional[Union[str, date]] = None,
    filter_names: Optional[str] = None,
    image_types: Optional[List[str]] = None,
    units: Optional[List[str]] = None,
    # → future params here, e.g. min_exptime, instrument, night_range, etc.
) -> Dict[str, List[Dict]]:
    """
    Core query: any combination of date, filter, types, units, ...
    """

    # normalize date
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

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
                f.name AS filter,
                n.date AS nightdate
            FROM
                "{tbl}" sf
            JOIN
                facility_unit u ON sf.unit_id = u.id
            JOIN
                survey_night n ON sf.night_id = n.id
            """

            if img_type in ("sci", "flat"):
                query += " LEFT JOIN facility_filter f ON sf.filter_id = f.id"

            # accumulate WHERE clauses
            clauses: List[str] = []
            params: List[Union[str, date]] = []

            if target_date:
                clauses.append("n.date = %s")
                params.append(target_date)
            if filter_names and img_type in ("sci", "flat"):
                clauses.append("f.name = %s")
                params.append(filter_names)
            if units:
                placeholders = ",".join(["%s"] * len(units))
                clauses.append(f"u.name IN ({placeholders})")
                params.extend(units)
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
                    }
                    for row in rows
                ]
            except Exception as e:
                print(f"Error querying {tbl}: {e}")
                results[img_type] = []

    conn.close()
    return results


class ImageQuery:
    """
    example:
    subset = (ImageQuery()
    .on_date(date.today())
    .for_units(["7DT01"])
    .of_types(["sci"])
    .fetch())
    """

    def __init__(self):
        self._params = {}

    def on_date(self, d: Union[str, date]):
        self._params["target_date"] = d
        return self

    def with_filter(self, filt: str):
        self._params["filter_names"] = filt
        return self

    def for_units(self, units: List[str]):
        self._params["units"] = units
        return self

    def of_types(self, types: List[str]):
        self._params["image_types"] = types
        return self

    # … add more fluent setters …

    def fetch(self) -> Dict[str, List[Dict]]:
        return get_image_files(**self._params)


# def get_image_files_by_date(
#     target_date: Union[str, date],
#     image_types: Optional[List[str]] = None,
#     units: Optional[List[str]] = None,
#     filters: Optional[List[str]] = None,
# ) -> Dict[str, List[Dict]]:
#     """
#     Get a list of image files corresponding to a given date.

#     Parameters:
#     -----------
#     target_date : str or datetime.date
#         The date to search for (format: 'YYYY-MM-DD' or datetime.date object)
#     image_types : list, optional
#         List of image types to filter by (e.g., ['scienceframe', 'biasframe', 'darkframe', 'flatframe'])
#         If None, returns all image types
#     units : list, optional
#         List of unit names to filter by (e.g., ['7DT01', '7DT02', ...])
#         If None, returns all units
#     filters : list, optional
#         List of filter names to filter by (e.g., ['r', 'g', 'i', 'z'])
#         If None, returns all filters

#     Returns:
#     --------
#     dict : Dictionary with image type as key and list of file information as value
#     """
#     # Convert target_date to date object if it's a string
#     if isinstance(target_date, str):
#         target_date = date.fromisoformat(target_date)

#     # Database connection
#     conn = psycopg.connect(
#         dbname=dbname,
#         user=user,
#         host=host,
#         port=port,
#         password=password,
#     )

#     results = {}

#     # Define the tables to query
#     tables = {
#         "scienceframe": "survey_scienceframe",
#         "biasframe": "survey_biasframe",
#         "darkframe": "survey_darkframe",
#         "flatframe": "survey_flatframe",
#     }

#     # Filter tables based on image_types parameter
#     if image_types:
#         tables = {k: v for k, v in tables.items() if k in image_types}

#     with conn.cursor() as cur:
#         for image_type, table_name in tables.items():
#             # Build the base query
#             query = f"""
#                 SELECT
#                     sf.id,
#                     sf.file_path,
#                     sf.original_filename,
#                     sf.unified_filename,
#                     sf.obstime,
#                     sf.exptime,
#                     sf.instrument,
#                     u.name AS unit_name,
#                     f.name AS filter_name,
#                     n.date
#                 FROM
#                     "{table_name}" sf
#                 JOIN survey_night n ON sf.night_id = n.id
#                 JOIN facility_unit u ON sf.unit_id = u.id
#             """

#             # Add filter join for science frames and flat frames
#             if image_type in ["scienceframe", "flatframe"]:
#                 query += " LEFT JOIN facility_filter f ON sf.filter_id = f.id"
#             else:
#                 query += " LEFT JOIN facility_filter f ON NULL"

#             # Add WHERE clause
#             conditions = ["n.date = %s"]
#             params = [target_date]

#             if units:
#                 placeholders = ",".join(["%s"] * len(units))
#                 conditions.append(f"u.name IN ({placeholders})")
#                 params.extend(units)

#             if filters and image_type in ["scienceframe", "flatframe"]:
#                 placeholders = ",".join(["%s"] * len(filters))
#                 conditions.append(f"f.name IN ({placeholders})")
#                 params.extend(filters)

#             query += " WHERE " + " AND ".join(conditions)
#             query += " ORDER BY sf.obstime"

#             try:
#                 cur.execute(query, params)
#                 rows = cur.fetchall()

#                 # Convert to list of dictionaries for easier handling
#                 file_list = []
#                 for row in rows:
#                     file_info = {
#                         "id": row[0],
#                         "file_path": row[1],
#                         "original_filename": row[2],
#                         "unified_filename": row[3],
#                         "obstime": row[4],
#                         "exptime": row[5],
#                         "instrument": row[6],
#                         "unit_name": row[7],
#                         "filter_name": row[8],
#                         "night_date": row[9],
#                         "full_path": row[1],  # This is the complete file path
#                     }
#                     file_list.append(file_info)

#                 results[image_type] = file_list

#             except Exception as e:
#                 print(f"Error querying {table_name}: {e}")
#                 results[image_type] = []

#     conn.close()
#     return results


# def get_image_files_summary(target_date: Union[str, date]) -> Dict[str, int]:
#     """
#     Get a summary of image files by type for a given date.

#     Parameters:
#     -----------
#     target_date : str or datetime.date
#         The date to search for

#     Returns:
#     --------
#     dict : Dictionary with image type as key and count as value
#     """
#     all_images = get_image_files_by_date(target_date)
#     return {image_type: len(files) for image_type, files in all_images.items()}


# def get_file_paths_by_date(
#     target_date: Union[str, date],
#     image_types: Optional[List[str]] = None,
#     units: Optional[List[str]] = None,
#     filters: Optional[List[str]] = None,
# ) -> Dict[str, List[str]]:
#     """
#     Get a list of full file paths corresponding to a given date.

#     Parameters:
#     -----------
#     target_date : str or datetime.date
#         The date to search for (format: 'YYYY-MM-DD' or datetime.date object)
#     image_types : list, optional
#         List of image types to filter by (e.g., ['scienceframe', 'biasframe', 'darkframe', 'flatframe'])
#         If None, returns all image types
#     units : list, optional
#         List of unit names to filter by (e.g., ['7DT01', '7DT02', ...])
#         If None, returns all units
#     filters : list, optional
#         List of filter names to filter by (e.g., ['r', 'g', 'i', 'z'])
#         If None, returns all filters

#     Returns:
#     --------
#     dict : Dictionary with image type as key and list of full file paths as value
#     """
#     results = get_image_files_by_date(target_date, image_types, units, filters)

#     # Extract just the file paths
#     path_results = {}
#     for image_type, files in results.items():
#         path_results[image_type] = [file_info["full_path"] for file_info in files]

#     return path_results


# def get_image_files_by_filter(
#     filter_name: str,
#     image_types: Optional[List[str]] = None,
#     units: Optional[List[str]] = None,
# ) -> Dict[str, List[Dict]]:
#     """
#     Get all image files in the database matching a given filter name,
#     regardless of date or other parameters.

#     Parameters:
#     -----------
#     filter_name : str
#         The filter name to search for (e.g., 'r', 'g', 'i', 'z').
#     image_types : list[str], optional
#         Subset of image types to include
#         (['scienceframe', 'biasframe', 'darkframe', 'flatframe']).
#         If None, searches all types.
#     units : list[str], optional
#         List of unit names to restrict by (e.g., ['7DT01', '7DT02']).
#         If None, returns all units.

#     Returns:
#     --------
#     dict[str, list[dict]]
#         Dictionary keyed by image type, each mapping to a list of file-info dicts.
#     """
#     conn = psycopg.connect(
#         dbname=dbname,
#         user=user,
#         host=host,
#         port=port,
#         password=password,
#     )
#     results: Dict[str, List[Dict]] = {}

#     tables = {
#         "scienceframe": "survey_scienceframe",
#         "biasframe": "survey_biasframe",
#         "darkframe": "survey_darkframe",
#         "flatframe": "survey_flatframe",
#     }
#     # Restrict to requested image types if given
#     if image_types:
#         tables = {k: v for k, v in tables.items() if k in image_types}

#     with conn.cursor() as cur:
#         for img_type, tbl in tables.items():
#             # Base SELECT
#             query = f"""
#                 SELECT
#                     sf.id,
#                     sf.file_path,
#                     sf.original_filename,
#                     sf.unified_filename,
#                     sf.obstime,
#                     sf.exptime,
#                     sf.instrument,
#                     u.name AS unit_name,
#                     f.name AS filter_name,
#                     n.date   AS night_date
#                 FROM "{tbl}" sf
#                 JOIN facility_unit u   ON sf.unit_id   = u.id
#                 JOIN survey_night  n   ON sf.night_id  = n.id
#             """
#             # Only science & flat have valid filter_id
#             if img_type in ("scienceframe", "flatframe"):
#                 query += " LEFT JOIN facility_filter f ON sf.filter_id = f.id"
#             else:
#                 query += " LEFT JOIN facility_filter f ON NULL"

#             # Build WHERE clauses
#             conditions = ["f.name = %s"]
#             params = [filter_name]

#             if units:
#                 placeholders = ", ".join("%s" for _ in units)
#                 conditions.append(f"u.name IN ({placeholders})")
#                 params.extend(units)

#             query += "\n WHERE " + " AND ".join(conditions)
#             query += "\n ORDER BY sf.obstime;"

#             try:
#                 cur.execute(query, params)
#                 rows = cur.fetchall()
#                 files = []
#                 for r in rows:
#                     files.append(
#                         {
#                             "id": r[0],
#                             "file_path": r[1],
#                             "original_filename": r[2],
#                             "unified_filename": r[3],
#                             "obstime": r[4],
#                             "exptime": r[5],
#                             "instrument": r[6],
#                             "unit_name": r[7],
#                             "filter_name": r[8],
#                             "night_date": r[9],
#                         }
#                     )
#                 results[img_type] = files
#             except Exception as e:
#                 print(f"Error querying {tbl}: {e}")
#                 results[img_type] = []

#     conn.close()
#     return results


# def get_image_files_by_unit(target_date: Union[str, date], unit: str) -> Dict[str, List[Dict]]:
#     """
#     Get image files for a specific unit on a given date.

#     Parameters:
#     -----------
#     target_date : str or datetime.date
#         The date to search for
#     unit : str
#         The unit name (e.g., '7DT01')

#     Returns:
#     --------
#     dict : Dictionary with image type as key and list of file information as value
#     """
#     return get_image_files_by_date(target_date, units=[unit])


# def get_science_frames_by_filter(target_date: Union[str, date], filter_name: str) -> List[Dict]:
#     """
#     Get science frames for a specific filter on a given date.

#     Parameters:
#     -----------
#     target_date : str or datetime.date
#         The date to search for
#     filter_name : str
#         The filter name (e.g., 'r', 'i', 'g', 'z')

#     Returns:
#     --------
#     list : List of science frame file information
#     """
#     results = get_image_files_by_date(target_date, image_types=["scienceframe"], filters=[filter_name])
#     return results.get("scienceframe", [])
