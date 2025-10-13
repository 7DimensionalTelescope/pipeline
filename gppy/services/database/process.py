import psycopg
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
import json
from contextlib import contextmanager
import os
from astropy.table import Table
import pandas as pd

# Import data classes
from .table import PipelineData
from .const import DB_PARAMS
from .utils import generate_id


class ProcessDBError(Exception):
    pass


class ProcessDB:
    """Database class for managing pipeline process data"""

    _connection = True
    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize with database parameters"""
        self.db_params = db_params or DB_PARAMS

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            if self._connection:
                conn = psycopg.connect(**self.db_params)
                yield conn
            else:
                return None
        except Exception as e:
            self._connection = False
            return None
        finally:
            if conn:
                conn.close()

    # ==================== PIPELINE DATA MANAGEMENT ====================

    def create_pipeline_data(self, pipeline_data: PipelineData) -> int:
        """Create a new pipeline data record or update existing one"""
        try:
            with self.get_connection() as conn:
                # Check if pipeline data already exists
                existing_pipeline = self._find_existing_pipeline_record(
                    pipeline_data.run_date, pipeline_data.data_type, pipeline_data.unit, pipeline_data.config_file
                )

                if existing_pipeline:
                    # Update existing record
                    pipeline_id = existing_pipeline
                    self.update_pipeline_data(pipeline_id, **pipeline_data.to_dict())
                    return pipeline_id
                else:
                    # Create new record
                    # Prepare parameters
                    params = pipeline_data.to_dict()

                    # Map fields to database columns and convert to JSON
                    if "run_date" in params:
                        params["date"] = params.pop("run_date")
                    if "bias" in params:
                        params["bias"] = params.pop("bias")
                    if "dark" in params:
                        params["dark"] = json.dumps(params.pop("dark") or [])
                    if "flat" in params:
                        params["flat"] = json.dumps(params.pop("flat") or [])

                    # Build query dynamically
                    columns = list(params.keys())
                    placeholders = [f"%({col})s" for col in columns]

                    # Add timestamp columns
                    columns.append("created_at")
                    columns.append("updated_at")
                    placeholders.append("CURRENT_TIMESTAMP")
                    placeholders.append("CURRENT_TIMESTAMP")

                    query = f"""
                        INSERT INTO pipeline_process 
                        ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                        RETURNING id;
                    """

                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        result = cur.fetchone()
                        conn.commit()

                        if result:
                            return result[0]
                        else:
                            raise ProcessDBError("Failed to get ID of created record")

        except Exception as e:
            raise ProcessDBError(f"Failed to create pipeline data: {e}")

    def read_pipeline_data(
        self,
        pipeline_id: Optional[int] = None,
        tag_id: Optional[str] = None,
        run_date: Optional[str] = None,
        data_type: Optional[str] = None,
        unit: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Union[Optional[PipelineData], List[PipelineData]]:
        """Read pipeline data records with optional filters"""
        try:
            with self.get_connection() as conn:
                # Build WHERE clause
                where_clauses = []
                params = {}

                if pipeline_id is not None:
                    where_clauses.append("id = %(pipeline_id)s")
                    params["pipeline_id"] = pipeline_id

                if tag_id:
                    where_clauses.append("tag_id = %(tag_id)s")
                    params["tag_id"] = tag_id

                if run_date:
                    # Handle date comparison more robustly
                    # Use a more flexible date comparison that handles different formats
                    where_clauses.append("date::date = %(run_date)s::date")
                    params["run_date"] = run_date

                if data_type:
                    where_clauses.append("data_type = %(data_type)s")
                    params["data_type"] = data_type

                if unit:
                    where_clauses.append("unit = %(unit)s")
                    params["unit"] = unit

                if status:
                    where_clauses.append("status = %(status)s")
                    params["status"] = status

                # Build query
                query = """
                    SELECT 
                        id, tag_id, date, data_type, obj, filt, unit, status, progress,
                        bias, dark, flat, warnings, errors, comments,
                        config_file, log_file, debug_file, comments_file, 
                        output_combined_frame_id, created_at, updated_at,
                        filename, param2, param3, param4, param5, param6, param7, param8, param9, param10
                    FROM pipeline_process
                """

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

                # If specific ID is requested, return single record
                if pipeline_id is not None:
                    query += " LIMIT 1"
                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        row = cur.fetchone()

                        if not row:
                            return None

                        # Create PipelineData object from row
                        pipeline_data = PipelineData.from_row(row)
                        return pipeline_data
                else:
                    # Return list of records
                    query += " ORDER BY date DESC, created_at DESC LIMIT %(limit)s OFFSET %(offset)s"
                    params["limit"] = limit
                    params["offset"] = offset

                    with conn.cursor() as cur:
                        cur.execute(query, params)
                        rows = cur.fetchall()

                        # Convert to PipelineData objects using from_row
                        result = [PipelineData.from_row(row) for row in rows]
                        return result

        except Exception as e:
            raise ProcessDBError(f"Failed to read pipeline data: {e}")

    def update_pipeline_data(self, pipeline_id: int, **kwargs) -> bool:
        """Update pipeline data record"""
        try:
            with self.get_connection() as conn:
                if not kwargs:
                    raise ProcessDBError("No fields to update")

                # Build SET clause
                set_clauses = []
                params = {"pipeline_id": pipeline_id}

                for key, value in kwargs.items():
                    # Map Python field names to database column names
                    if key == "bias":
                        set_clauses.append("bias_exists = %(bias)s")
                        params["bias"] = value
                    elif key == "dark":
                        set_clauses.append("dark_filters = %(dark)s")
                        params["dark"] = json.dumps(value) if isinstance(value, list) else json.dumps([])
                    elif key == "flat":
                        set_clauses.append("flat_filters = %(flat)s")
                        params["flat"] = json.dumps(value) if isinstance(value, list) else json.dumps([])
                    elif key == "run_date":
                        set_clauses.append("date = %(run_date)s")
                        params["run_date"] = value
                    else:
                        set_clauses.append(f"{key} = %({key})s")
                        params[key] = value

                # Add updated_at timestamp
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")

                query = f"""
                    UPDATE pipeline_process 
                    SET {', '.join(set_clauses)}
                    WHERE id = %(pipeline_id)s
                """

                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise ProcessDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to update pipeline data: {e}")

    def delete_pipeline_data(self, pipeline_id: int) -> bool:
        """Delete a pipeline data record"""
        try:
            with self.get_connection() as conn:
                query = "DELETE FROM pipeline_process WHERE id = %s"

                with conn.cursor() as cur:
                    cur.execute(query, (pipeline_id,))
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise ProcessDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to delete pipeline data: {e}")

    def _find_existing_pipeline_record(
        self, run_date: str, data_type: str, unit: str, config_file: Optional[str] = None
    ) -> Optional[int]:
        """Find existing pipeline record by configuration parameters"""
        try:
            # Use the updated read_pipeline_data method with filters
            pipeline_records = self.read_pipeline_data(run_date=run_date, data_type=data_type, unit=unit, limit=1)

            if isinstance(pipeline_records, list) and pipeline_records:
                # Filter by config_file if specified
                if config_file:
                    for record in pipeline_records:
                        if record.config_file == config_file:
                            return record.id
                    return None
                else:
                    return pipeline_records[0].id
            else:
                return None

        except Exception as e:
            raise ProcessDBError(f"Failed to find existing pipeline record: {e}")

    def add_warning(self, pipeline_id: int, count: int = 1) -> bool:
        """Add warning count to pipeline data record"""
        try:
            with self.get_connection() as conn:
                query = """
                    UPDATE pipeline_process 
                    SET warnings = warnings + %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """

                with conn.cursor() as cur:
                    cur.execute(query, (count, pipeline_id))
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise ProcessDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to add warning: {e}")

    def add_error(self, pipeline_id: int, count: int = 1) -> bool:
        """Add error count to pipeline data record"""
        try:
            with self.get_connection() as conn:
                query = """
                    UPDATE pipeline_process 
                    SET errors = errors + %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """

                with conn.cursor() as cur:
                    cur.execute(query, (count, pipeline_id))
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise ProcessDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to add error: {e}")

    def delete_pipeline_cascade(self, pipeline_id: int) -> bool:
        """Delete pipeline data (QA data should be handled separately)"""
        try:
            # Only delete pipeline data - QA data should be handled by ImageDB
            return self.delete_pipeline_data(pipeline_id)

        except Exception as e:
            raise ProcessDBError(f"Failed to delete pipeline: {e}")

    # ==================== EXPORT OPERATIONS ====================

    def export_pipeline_data_to_table(self) -> pd.DataFrame:
        """Export pipeline data to pandas DataFrame"""
        try:
            with self.get_connection() as conn:
                # Get all pipeline data
                query = """
                    SELECT 
                        id, tag_id, date, data_type, obj, filt, unit, status, progress,
                        bias, dark, flat, warnings, errors, comments,
                        config_file, log_file, debug_file, comments_file, 
                        output_combined_frame_id, created_at, updated_at,
                        filename, param2, param3, param4, param5, param6, param7, param8, param9, param10
                    FROM pipeline_process
                    ORDER BY date DESC, created_at DESC
                """

                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

                    if not rows:
                        print(f"No pipeline data found to export")
                        return pd.DataFrame()

                    # Get column names
                    columns = [desc[0] for desc in cur.description]

                    # Create pandas DataFrame
                    df = pd.DataFrame(rows, columns=columns)
                    return df
        except Exception as e:
            raise ProcessDBError(f"Failed to export pipeline data to table: {e}")

    def export_pipeline_data_to_csv(self, filename: str) -> bool:
        """Export pipeline data to CSV file"""
        try:
            # Get pipeline data using the table function
            df = self.export_pipeline_data_to_table()

            if df.empty:
                print(f"No pipeline data found to export")
                return False

            # Write to CSV file using pandas
            df.to_csv(filename, index=False)

            print(f"Exported {len(df)} pipeline records to {filename}")
            return True

        except Exception as e:
            raise ProcessDBError(f"Failed to export pipeline data: {e}")

    # ==================== CLEANUP OPERATIONS ====================

    def clear_pipeline_data(self) -> bool:
        """Clear all pipeline data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM pipeline_process")
                    pipeline_deleted = cur.rowcount
                    conn.commit()

                    print(f"Cleared {pipeline_deleted} pipeline records")
                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to clear pipeline data: {e}")
