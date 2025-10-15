import psycopg
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
import json
from contextlib import contextmanager
import os
import pandas as pd

# Import data classes
from .table import QAData
from .const import DB_PARAMS
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
                    where_clauses.append("pipeline_id = %(pipeline_id)s")
                    params["pipeline_id"] = pipeline_id

                if qa_type:
                    where_clauses.append("qa_type = %(qa_type)s")
                    params["qa_type"] = qa_type

                if filt:
                    where_clauses.append("filter = %(filt)s")
                    params["filt"] = filt

                # Build query
                query = """
                    SELECT 
                        id, qa_id, qa_type, imagetyp, filter, clipmed, clipstd,
                        clipmin, clipmax, nhotpix, ntotpix, seeing, ellipticity,
                        rotang1, astrometric_offset, skyval, skysig, zp_auto, ezp_auto,
                        ul5_5, stdnumb, created_at, updated_at, pipeline_id_id,
                        uniform, sigmean, edgevar, trimmed, exptime, qa6, qa7, qa8, filename, sanity
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

    def _find_existing_qa_data(self, qa_data: QAData) -> Optional[str]:
        """Find existing QA data record by key fields"""
        try:
            with self.get_connection() as conn:
                # Build WHERE clause based on key identifying fields
                where_clauses = []
                params = {}

                if qa_data.qa_id:
                    where_clauses.append("qa_id = %(qa_id)s")
                    params["qa_id"] = qa_data.qa_id

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

                # Need at least some identifying fields to find existing record
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

    def get_qa_summary_for_pipeline(self, pipeline_id: int) -> Dict[str, Any]:
        """Get QA summary statistics for a specific pipeline"""
        try:
            qa_data = self.read_qa_data(pipeline_id=pipeline_id)
            if not isinstance(qa_data, list):
                qa_data = [qa_data] if qa_data else []

            if not qa_data:
                return {"total_count": 0, "types": [], "by_type": {}, "quality_metrics": {}}

            # Group by type and calculate metrics
            qa_by_type = {}
            quality_metrics = {"total_seeing": 0, "total_ellipticity": 0, "total_skyval": 0, "valid_metrics": 0}

            for qa in qa_data:
                qa_type = qa.qa_type or "unknown"
                if qa_type not in qa_by_type:
                    qa_by_type[qa_type] = []
                qa_by_type[qa_type].append(qa)

                # Aggregate quality metrics
                if qa.seeing is not None:
                    quality_metrics["total_seeing"] += qa.seeing
                    quality_metrics["valid_metrics"] += 1
                if qa.ellipticity is not None:
                    quality_metrics["total_ellipticity"] += qa.ellipticity
                if qa.skyval is not None:
                    quality_metrics["total_skyval"] += qa.skyval

            # Calculate averages
            if quality_metrics["valid_metrics"] > 0:
                quality_metrics["avg_seeing"] = quality_metrics["total_seeing"] / quality_metrics["valid_metrics"]
                quality_metrics["avg_ellipticity"] = (
                    quality_metrics["total_ellipticity"] / quality_metrics["valid_metrics"]
                )
                quality_metrics["avg_skyval"] = quality_metrics["total_skyval"] / quality_metrics["valid_metrics"]

            return {
                "total_count": len(qa_data),
                "types": list(qa_by_type.keys()),
                "by_type": {k: len(v) for k, v in qa_by_type.items()},
                "quality_metrics": quality_metrics,
            }

        except Exception as e:
            return {"total_count": 0, "types": [], "by_type": {}, "quality_metrics": {}, "error": str(e)}

    def export_qa_data_to_table(self) -> pd.DataFrame:
        """Export QA data to pandas DataFrame"""
        return self.export_to_table("pipeline_qa", "created_at DESC")

    def export_qa_data_to_csv(self, filename: str) -> bool:
        """Export QA data to CSV file"""
        return self.export_to_csv("pipeline_qa", filename, "created_at DESC")

    def clear_qa_data(self) -> bool:
        """Clear all QA data"""
        return self.clear_table("pipeline_qa")
