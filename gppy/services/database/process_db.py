import psycopg
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
import json
from contextlib import contextmanager
import os

# Import data classes and ImageDB
from .table import PipelineData
from .image_db import ImageDB
from .const import DB_PARAMS
from .utils import generate_id


class ProcessDBError(Exception):
    pass


class ProcessDB(ImageDB):
    """Database class for managing pipeline process data and associated QA data"""

    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize with database parameters"""
        super().__init__(db_params)

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
                        INSERT INTO pipeline_pipelinedata 
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
                    FROM pipeline_pipelinedata
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
                    UPDATE pipeline_pipelinedata 
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
                query = "DELETE FROM pipeline_pipelinedata WHERE id = %s"

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
                    UPDATE pipeline_pipelinedata 
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
                    UPDATE pipeline_pipelinedata 
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

    # ==================== COMBINED OPERATIONS ====================

    def get_pipeline_with_qa(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """Get pipeline data with associated QA data"""
        try:
            # Get pipeline data
            pipeline_record = self.read_pipeline_data(pipeline_id=pipeline_id)
            if not pipeline_record:
                return None

            # Get associated QA data using inherited method
            qa_data = self.read_qa_data(pipeline_id_id=pipeline_id)
            if isinstance(qa_data, list):
                qa_list = qa_data
            else:
                qa_list = [qa_data] if qa_data else []

            return {"pipeline": pipeline_record, "qa_data": qa_list, "qa_count": len(qa_list)}

        except Exception as e:
            raise ProcessDBError(f"Failed to get pipeline with QA data: {e}")

    def create_pipeline_with_qa(self, pipeline_data: PipelineData, qa_data_list: List) -> Dict[str, int]:
        """Create pipeline data with associated QA data in a single transaction"""
        try:
            # Create pipeline first
            pipeline_id = self.create_pipeline_data(pipeline_data)

            # Create QA data with pipeline_id reference using inherited method
            qa_ids = []
            for qa_data in qa_data_list:
                qa_data.pipeline_id_id = pipeline_id
                qa_id = self.create_qa_data(qa_data)
                qa_ids.append(qa_id)

            return {"pipeline_id": pipeline_id, "qa_ids": qa_ids, "total_qa_records": len(qa_ids)}

        except Exception as e:
            raise ProcessDBError(f"Failed to create pipeline with QA data: {e}")

    def update_pipeline_and_qa(
        self, pipeline_id: int, pipeline_updates: Dict[str, Any], qa_updates: List[Dict[str, Any]]
    ) -> bool:
        """Update pipeline data and associated QA data in a single transaction"""
        try:
            # Update pipeline data
            if pipeline_updates:
                self.update_pipeline_data(pipeline_id, **pipeline_updates)

            # Update QA data using inherited method
            for qa_update in qa_updates:
                qa_id = qa_update.pop("qa_id", None)
                if qa_id and qa_update:
                    self.update_qa_data(qa_id, **qa_update)

            return True

        except Exception as e:
            raise ProcessDBError(f"Failed to update pipeline and QA data: {e}")

    def delete_pipeline_cascade(self, pipeline_id: int) -> bool:
        """Delete pipeline data and all associated QA data"""
        try:
            # Delete QA data first (due to foreign key constraints) using inherited method
            qa_data = self.read_qa_data(pipeline_id_id=pipeline_id)
            if isinstance(qa_data, list):
                for qa in qa_data:
                    self.delete_qa_data(qa.qa_id)

            # Delete pipeline data
            return self.delete_pipeline_data(pipeline_id)

        except Exception as e:
            raise ProcessDBError(f"Failed to delete pipeline cascade: {e}")

    # ==================== EXPORT OPERATIONS ====================

    def export_pipeline_data_to_ecsv(self, filename: str) -> bool:
        """Export pipeline data to ECSV file"""
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
                    FROM pipeline_pipelinedata
                    ORDER BY date DESC, created_at DESC
                """

                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

                    if not rows:
                        print(f"No pipeline data found to export")
                        return False

                    # Get column names
                    columns = [desc[0] for desc in cur.description]

                    # Write to ECSV file
                    with open(filename, "w") as f:
                        # Write ECSV header
                        f.write("# %ECSV 1.0\n")
                        f.write("# ---\n")
                        f.write("# datatype: table\n")
                        f.write(f"# colcount: {len(columns)}\n")

                        # Write column definitions
                        for i, col in enumerate(columns):
                            f.write(f"# col{str(i+1).zfill(2)}: name: {col}\n")

                        # Write data
                        f.write("# ---\n")
                        f.write(",".join(columns) + "\n")

                        for row in rows:
                            # Convert each value to string, handling None values
                            row_str = []
                            for val in row:
                                if val is None:
                                    row_str.append("")
                                elif isinstance(val, (dict, list)):
                                    row_str.append(json.dumps(val))
                                else:
                                    row_str.append(str(val))
                            f.write(",".join(row_str) + "\n")

                    print(f"Exported {len(rows)} pipeline records to {filename}")
                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to export pipeline data: {e}")

    def export_to_ecsv(self, base_filename: str) -> Dict[str, str]:
        """
        Export database tables to ECSV files.

        Args:
            base_filename: Base filename without extension (e.g., "2025-01-27_unit1")

        Returns:
            Dict with paths to the exported files:
            {
                'pipeline_data': 'XX_process.ecsv',
                'qa_data': 'YY_qa.ecsv'
            }
        """
        try:
            # Generate filenames
            if base_filename.endswith("_process.ecsv"):
                pipeline_filename = base_filename
                qa_filename = base_filename.replace("_process.ecsv", "_qa.ecsv")
            elif base_filename.endswith(".ecsv"):
                pipeline_filename = base_filename.replace(".ecsv", "_process.ecsv")
                qa_filename = base_filename.replace(".ecsv", "_qa.ecsv")
            else:
                pipeline_filename = f"{base_filename}_process.ecsv"
                qa_filename = f"{base_filename}_qa.ecsv"

            # Export pipeline data
            pipeline_data = self.export_pipeline_data_to_ecsv(pipeline_filename)

            # Export QA data using inherited method
            qa_data = self.export_qa_data_to_ecsv(qa_filename)

            return {"pipeline_data": pipeline_filename, "qa_data": qa_filename}

        except Exception as e:
            raise ProcessDBError(f"Failed to export database to ECSV: {e}")

    # ==================== CLEANUP OPERATIONS ====================

    def clear_pipeline_data(self) -> bool:
        """Clear all pipeline data"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM pipeline_pipelinedata")
                    pipeline_deleted = cur.rowcount
                    conn.commit()

                    print(f"Cleared {pipeline_deleted} pipeline records")
                    return True

        except Exception as e:
            raise ProcessDBError(f"Failed to clear pipeline data: {e}")

    def clear_database(self) -> bool:
        """Clear all data from pipeline tables"""
        try:
            # Clear QA data first (due to foreign key constraints) using inherited method
            qa_deleted = self.clear_qa_data()

            # Clear pipeline data
            pipeline_deleted = self.clear_pipeline_data()

            print(f"Cleared {qa_deleted} QA records and {pipeline_deleted} pipeline records")
            return True

        except Exception as e:
            raise ProcessDBError(f"Failed to clear database: {e}")
