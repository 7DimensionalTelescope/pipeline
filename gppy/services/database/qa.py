import psycopg
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
import json
from contextlib import contextmanager
import os
import pandas as pd

# Import data classes
from .table import QAData, PipelineData
from .base import BaseDatabase, DatabaseError


class QADBError(DatabaseError):
    pass


class QADB(BaseDatabase):
    """Database class for managing image QA data"""

    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize with database parameters"""
        super().__init__(db_params)

    def create_qa_data(self, qa_data: QAData) -> str:
        """Create a new QA data record or update existing one"""
        try:
            with self.get_connection() as conn:
                # Check if QA data already exists
                existing_qa = self._find_existing_qa_data(qa_data)

                if existing_qa:
                    # Update existing record
                    qa_id = existing_qa
                    qa_dict = qa_data.to_dict()
                    qa_dict["qa_id"] = qa_id
                    self.update_qa_data(**qa_dict)
                    return qa_id
                else:
                    # Create new record
                    # Prepare parameters - filter out None values
                    params = {k: v for k, v in qa_data.to_dict().items() if v is not None}

                    # Build query dynamically
                    columns = list(params.keys())
                    placeholders = [f"%({col})s" for col in columns]

                    # Add timestamp columns
                    columns.append("created_at")
                    columns.append("updated_at")
                    placeholders.append("CURRENT_TIMESTAMP")
                    placeholders.append("CURRENT_TIMESTAMP")

                    query = f"""
                        INSERT INTO pipeline_qa 
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                        RETURNING qa_id;
                    """

                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        result = cur.fetchone()
                        conn.commit()

                        if result:
                            return result[0]
                        else:
                            raise QADBError("Failed to get qa_id of created QA record")

        except Exception as e:
            raise QADBError(f"Failed to create QA data: {e}")

    def read_qa_data(
        self,
        qa_id: Optional[str] = None,
        pipeline_id: Optional[int] = None,
        qa_type: Optional[str] = None,
        filt: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Union[Optional[QAData], List[QAData]]:
        """Read QA data records with optional filters"""
        try:
            with self.get_connection() as conn:
                # Build WHERE clause
                where_clauses = []
                params = {}

                if qa_id:
                    where_clauses.append("qa_id = %(qa_id)s")
                    params["qa_id"] = qa_id

                if pipeline_id is not None:
                    where_clauses.append("pipeline_id_id = %(pipeline_id)s")
                    params["pipeline_id_id"] = pipeline_id

                if qa_type:
                    where_clauses.append("qa_type = %(qa_type)s")
                    params["qa_type"] = qa_type

                if filt:
                    where_clauses.append("filter = %(filt)s")
                    params["filt"] = filt

                # Build query
                query = """
                    SELECT 
                        id, qa_id, qa_type, imagetyp, filter, date_obs, clipmed, clipstd,
                        clipmin, clipmax, nhotpix, ntotpix, seeing, rotang,
                        peeing, ptnoff, visible, skyval, skysig, zp_auto, ezp_auto,
                        ul5_5, stdnumb, created_at, updated_at, pipeline_id_id,
                        edgevar, exptime, filename, sanity, sigmean, trimmed, unmatch,
                        rsep_rms, rsep_q2, uniform, awincrmn, ellipmn, rsep_p95, pa_align,
                        q_desc, eye_insp
                    FROM pipeline_qa
                """

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                # If specific qa_id is requested, return single record
                if qa_id:
                    query += " LIMIT 1"
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        row = cur.fetchone()

                        if not row:
                            return None

                        # Convert to QAData object using from_row
                        qa_data = QAData.from_row(row)
                        return qa_data
                else:
                    # Return list of records
                    query += " ORDER BY created_at DESC LIMIT %(limit)s OFFSET %(offset)s"
                    params["limit"] = limit
                    params["offset"] = offset

                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        rows = cur.fetchall()

                        # Convert to QAData objects using from_row
                        result = [QAData.from_row(row) for row in rows]
                        return result

        except Exception as e:
            raise QADBError(f"Failed to read QA data: {e}")

    def update_qa_data(self, qa_id: str, **kwargs) -> bool:
        """Update QA data record"""
        try:
            with self.get_connection() as conn:
                if not kwargs:
                    raise QADBError("No fields to update")

                # Filter out None values to prevent constraint violations
                update_data = {k: v for k, v in kwargs.items() if v is not None}

                if not update_data:
                    raise QADBError("No non-None fields to update")

                # Build SET clause
                set_clauses = []
                params = {"qa_id": qa_id}

                for key, value in update_data.items():
                    set_clauses.append(f"{key} = %({key})s")
                    params[key] = value

                # Add updated_at timestamp
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")

                query = f"""
                    UPDATE pipeline_qa 
                    SET {', '.join(set_clauses)}
                    WHERE qa_id = %(qa_id)s
                """

                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise QADBError(f"No QA record found with ID {qa_id}")

                    return True

        except Exception as e:
            raise QADBError(f"Failed to update QA data: {e}")

    def delete_qa_data(self, qa_id: str) -> bool:
        """Delete a QA data record"""
        try:
            with self.get_connection() as conn:
                query = "DELETE FROM pipeline_qa WHERE qa_id = %s"

                with conn.cursor() as cur:
                    cur.execute(query, (qa_id,))
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise QADBError(f"No QA record found with ID {qa_id}")

                    return True

        except Exception as e:
            raise QADBError(f"Failed to delete QA data: {e}")

    def delete_qa_data_by_pipeline_id(self, pipeline_id: int) -> bool:
        """Delete all QA data records for a specific pipeline ID"""
        try:
            with self.get_connection() as conn:
                query = "DELETE FROM pipeline_qa WHERE pipeline_id_id = %s"

                with conn.cursor() as cur:
                    cur.execute(query, (pipeline_id,))
                    rows_affected = cur.rowcount
                    conn.commit()

                    return True

        except Exception as e:
            raise QADBError(f"Failed to delete QA data by pipeline ID: {e}")

    def _find_existing_qa_data(self, qa_data: QAData) -> Optional[str]:
        """Find existing QA data record by key fields including filename"""
        try:
            with self.get_connection() as conn:
                # Build WHERE clause based on key identifying fields
                # Priority: Check by filename first (most specific), then by other fields
                where_clauses = []
                params = {}

                # Priority: Check by filename first (most specific identifier to prevent duplicates)
                # Don't check by qa_id when looking for existing records (qa_id is generated new each time)
                # Only use qa_id if explicitly looking for a specific record

                if qa_data.qa_id:
                    # If qa_id is provided and we're looking for a specific record, use it
                    where_clauses.append("qa_id = %(qa_id)s")
                    params["qa_id"] = qa_data.qa_id

                # Always check by filename if provided (primary duplicate prevention)
                if qa_data.filename:
                    where_clauses.append("filename = %(filename)s")
                    params["filename"] = qa_data.filename

                if qa_data.pipeline_id_id is not None:
                    where_clauses.append("pipeline_id_id = %(pipeline_id_id)s")
                    params["pipeline_id_id"] = qa_data.pipeline_id_id

                if qa_data.qa_type:
                    where_clauses.append("qa_type = %(qa_type)s")
                    params["qa_type"] = qa_data.qa_type

                if qa_data.filter:
                    where_clauses.append("filter = %(filter)s")
                    params["filter"] = qa_data.filter

                if qa_data.imagetyp:
                    where_clauses.append("imagetyp = %(imagetyp)s")
                    params["imagetyp"] = qa_data.imagetyp

                # If filename is provided, require pipeline_id, qa_type, imagetyp for a match
                # This ensures we find the correct existing record by filename
                if qa_data.filename:
                    # Require pipeline_id, qa_type, and imagetyp along with filename
                    # Don't require qa_id (it's generated new each time)
                    if qa_data.pipeline_id_id is None or not qa_data.qa_type or not qa_data.imagetyp:
                        # Not enough info to find existing record by filename
                        return None
                    # Remove qa_id from the check when filename is provided (to find existing records)
                    # Keep only filename, pipeline_id, qa_type, imagetyp
                    where_clauses = [clause for clause in where_clauses if not clause.startswith("qa_id =")]
                    if "qa_id" in params:
                        del params["qa_id"]
                else:
                    # Without filename, need at least some identifying fields
                    if not where_clauses:
                        return None

                where_clause = " AND ".join(where_clauses)

                query = f"""
                    SELECT qa_id FROM pipeline_qa
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT 1
                """

                with conn.cursor() as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()

                    if result:
                        return result[0]
                    else:
                        return None

        except Exception as e:
            # If there's an error finding existing record, return None to create new
            return None

    def export_qa_data_to_table(self) -> pd.DataFrame:
        """Export QA data to pandas DataFrame"""
        return self.export_to_table("pipeline_qa", "created_at DESC")

    def export_qa_data_to_csv(self, filename: str) -> bool:
        """Export QA data to CSV file"""
        return self.export_to_csv("pipeline_qa", filename, "created_at DESC")

    def clear_qa_data(self) -> bool:
        """Clear all QA data"""
        return self.clear_table("pipeline_qa")

    def get_enhanced_qa_records(
        self, qa_type: str, param: Optional[str] = None, date_min: Optional[str] = None, date_max: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get QA records enhanced with pipeline properties in a single query.
        Much faster than the for loop approach.
        Only selects necessary columns to optimize query performance.

        Args:
            qa_type: QA type to filter by ('bias', 'dark', 'flat', 'science')
            param: Optional parameter name to filter (only return this parameter)
            date_min: Optional minimum date filter (YYYY-MM-DD)
            date_max: Optional maximum date filter (YYYY-MM-DD)

        Returns:
            List of dictionaries with QA data enhanced with pipeline properties
        """
        try:
            # Validate qa_type
            valid_types = {"bias", "dark", "flat", "science"}
            if qa_type not in valid_types:
                return []

            with self.get_connection() as conn:
                # Base columns always needed for filtering/grouping
                base_columns = [
                    "qa.qa_type",
                    "qa.created_at",
                    "qa.date_obs",
                    "p.unit",
                    "p.filt",
                    "p.obj",
                    "p.date as run_date",
                ]

                # If param is specified, only select that parameter column (optimized query)
                if param:
                    # Map parameter names to database column names
                    param_to_column = {
                        "clipmed": "qa.clipmed",
                        "clipstd": "qa.clipstd",
                        "clipmin": "qa.clipmin",
                        "clipmax": "qa.clipmax",
                        "nhotpix": "qa.nhotpix",
                        "ntotpix": "qa.ntotpix",
                        "seeing": "qa.seeing",
                        "rotang": "qa.rotang",
                        "peeing": "qa.peeing",
                        "ptnoff": "qa.ptnoff",
                        "visible": "qa.visible",
                        "skyval": "qa.skyval",
                        "skysig": "qa.skysig",
                        "zp_auto": "qa.zp_auto",
                        "ezp_auto": "qa.ezp_auto",
                        "ul5_5": "qa.ul5_5",
                        "stdnumb": "qa.stdnumb",
                        "edgevar": "qa.edgevar",
                        "exptime": "qa.exptime",
                        "sigmean": "qa.sigmean",
                        "unmatch": "qa.unmatch",
                        "rsep_rms": "qa.rsep_rms",
                        "rsep_q2": "qa.rsep_q2",
                        "uniform": "qa.uniform",
                        "awincrmn": "qa.awincrmn",
                        "ellipmn": "qa.ellipmn",
                        "rsep_p95": "qa.rsep_p95",
                        "pa_align": "qa.pa_align",
                    }

                    # Check if param exists in mapping
                    if param in param_to_column:
                        # Only select base columns + the requested parameter (optimized)
                        selected_columns = base_columns + [param_to_column[param]]
                        # For dark type, also include exptime
                        if qa_type == "dark" and param != "exptime":
                            selected_columns.append("qa.exptime")
                    else:
                        # If param not found, return empty (invalid parameter)
                        return []
                else:
                    # When param is not specified, select all columns (full data)
                    selected_columns = [
                        "qa.id",
                        "qa.qa_id",
                        "qa.qa_type",
                        "qa.imagetyp",
                        "qa.filter",
                        "qa.date_obs",
                        "qa.clipmed",
                        "qa.clipstd",
                        "qa.clipmin",
                        "qa.clipmax",
                        "qa.nhotpix",
                        "qa.ntotpix",
                        "qa.seeing",
                        "qa.rotang",
                        "qa.peeing",
                        "qa.ptnoff",
                        "qa.visible",
                        "qa.skyval",
                        "qa.skysig",
                        "qa.zp_auto",
                        "qa.ezp_auto",
                        "qa.ul5_5",
                        "qa.stdnumb",
                        "qa.created_at",
                        "qa.updated_at",
                        "qa.pipeline_id_id",
                        "qa.edgevar",
                        "qa.exptime",
                        "qa.filename",
                        "qa.sanity",
                        "qa.sigmean",
                        "qa.trimmed",
                        "qa.unmatch",
                        "qa.rsep_rms",
                        "qa.rsep_q2",
                        "qa.uniform",
                        "qa.awincrmn",
                        "qa.ellipmn",
                        "qa.rsep_p95",
                        "qa.pa_align",
                        "qa.q_desc",
                        "qa.eye_insp",
                        "p.unit",
                        "p.filt",
                        "p.obj",
                        "p.date as run_date",
                    ]

                # Build WHERE clause with date filters
                where_clauses = ["qa.qa_type = %s"]
                params = [qa_type]

                if date_min:
                    where_clauses.append("DATE(qa.created_at) >= %s")
                    params.append(date_min)

                if date_max:
                    where_clauses.append("DATE(qa.created_at) <= %s")
                    params.append(date_max)

                # Query only necessary columns (no LIMIT - fetch all matching records)
                query = f"""
                    SELECT 
                        {', '.join(selected_columns)}
                    FROM pipeline_qa qa
                    LEFT JOIN pipeline_process p ON qa.pipeline_id_id = p.id
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY qa.created_at DESC
                """

                with conn.cursor() as cur:
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()

                    enhanced_records = []
                    for row in rows:
                        if param:
                            # When param is specified, only return essential fields + the parameter
                            record_dict = {
                                "qa_type": row[0],
                                "created_at": row[1],
                                "date_obs": row[2],
                                "unit": row[3],
                                "filter": row[4],
                                "object": row[5],
                                "run_date": row[6],
                                param: row[7] if len(row) > 7 else None,
                            }
                            # For dark type, also include exptime if it was selected
                            if qa_type == "dark" and len(row) > 8:
                                record_dict["exptime"] = row[8] if row[8] is not None else None
                        else:
                            # Create full enhanced record dictionary
                            record_dict = {
                                "id": row[0],
                                "qa_id": row[1],
                                "qa_type": row[2],
                                "imagetyp": row[3],
                                "filter": row[4],
                                "date_obs": row[5],
                                "clipmed": row[6],
                                "clipstd": row[7],
                                "clipmin": row[8],
                                "clipmax": row[9],
                                "nhotpix": row[10],
                                "ntotpix": row[11],
                                "seeing": row[12],
                                "rotang": row[13],
                                "peeing": row[14],
                                "ptnoff": row[15],
                                "visible": row[16],
                                "skyval": row[17],
                                "skysig": row[18],
                                "zp_auto": row[19],
                                "ezp_auto": row[20],
                                "ul5_5": row[21],
                                "stdnumb": row[22],
                                "created_at": row[23],
                                "updated_at": row[24],
                                "pipeline_id_id": row[25],
                                "edgevar": row[26],
                                "exptime": row[27],
                                "filename": row[28],
                                "sanity": row[29],
                                "sigmean": row[30],
                                "trimmed": row[31],
                                "unmatch": row[32],
                                "rsep_rms": row[33],
                                "rsep_q2": row[34],
                                "uniform": row[35],
                                "awincrmn": row[36],
                                "ellipmn": row[37],
                                "rsep_p95": row[38],
                                "pa_align": row[39],
                                "q_desc": row[40],
                                "eye_insp": row[41],
                                # Enhanced with pipeline properties
                                "unit": row[42],
                                "filter": row[43]
                                or row[4],  # This will override the QA filter if pipeline filter exists
                                "object": row[44],
                                "run_date": row[45],
                            }
                        enhanced_records.append(record_dict)

                    return enhanced_records

        except Exception as e:
            raise QADBError(f"Failed to get enhanced QA records: {e}")
