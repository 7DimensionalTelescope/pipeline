import psycopg
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
import json
from contextlib import contextmanager
import os

# Import data classes from table module
from .table import PipelineData, QAData
from .const import DB_PARAMS
from .utils import generate_id


class PipelineDatabase:
    """Main class for pipeline database operations"""

    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize with database parameters"""
        self.db_params = db_params or DB_PARAMS

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg.connect(**self.db_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise PipelineDBError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()

    def create_pipeline_data(self, pipeline_data: PipelineData) -> int:
        """Create a new pipeline data record"""
        try:
            with self.get_connection() as conn:
                # Prepare parameters
                params = pipeline_data.to_dict()

                # Map fields to database columns and convert to JSON
                if "run_date" in params:
                    params["date"] = params.pop("run_date")
                if "bias" in params:
                    params["bias_exists"] = params.pop("bias")
                if "dark" in params:
                    params["dark_filters"] = json.dumps(params.pop("dark") or [])
                if "flat" in params:
                    params["flat_filters"] = json.dumps(params.pop("flat") or [])

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
                        raise PipelineDBError("Failed to get ID of created record")

        except Exception as e:
            raise PipelineDBError(f"Failed to create pipeline data: {e}")

    def read_pipeline_data(self, pipeline_id: int) -> Optional[PipelineData]:
        """Read pipeline data record by ID"""
        try:
            with self.get_connection() as conn:
                # Build query
                query = """
                    SELECT 
                        id, tag_id, date, data_type, obj, filt, unit, status, progress,
                        bias_exists, dark_filters, flat_filters, warnings, errors, comments,
                        config_file, log_file, debug_file, comments_file, 
                        output_combined_frame_id, created_at, updated_at,
                        param1, param2, param3, param4, param5, param6, param7, param8, param9, param10
                    FROM pipeline_pipelinedata
                    WHERE id = %s
                """

                with conn.cursor() as cur:
                    cur.execute(query, (pipeline_id,))
                    row = cur.fetchone()

                    if not row:
                        return None

                    # Create PipelineData object from row
                    pipeline_data = PipelineData.from_row(row)

                    return pipeline_data

        except Exception as e:
            raise PipelineDBError(f"Failed to read pipeline data: {e}")

    def update_pipeline_data(self, pipeline_id: int, **kwargs) -> bool:
        """Update pipeline data record"""
        try:
            with self.get_connection() as conn:
                if not kwargs:
                    raise PipelineDBError("No fields to update")

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
                        raise PipelineDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to update pipeline data: {e}")

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
                        raise PipelineDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to delete pipeline data: {e}")

    def create_qa_data(self, qa_data: QAData) -> int:
        """Create a new QA data record"""
        try:
            with self.get_connection() as conn:
                # Prepare parameters
                params = qa_data.to_dict()

                # Build query dynamically
                columns = list(params.keys())
                placeholders = [f"%({col})s" for col in columns]

                # Add timestamp columns
                columns.append("created_at")
                columns.append("updated_at")
                placeholders.append("CURRENT_TIMESTAMP")
                placeholders.append("CURRENT_TIMESTAMP")

                query = f"""
                    INSERT INTO pipeline_qadata 
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
                        raise PipelineDBError("Failed to get ID of created QA record")

        except Exception as e:
            raise PipelineDBError(f"Failed to create QA data: {e}")

    def read_qa_data(
        self,
        qa_id: Optional[str] = None,
        pipeline_id_id: Optional[int] = None,
        qa_type: Optional[str] = None,
        filter_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[QAData]:
        """Read QA data records with optional filters"""
        try:
            with self.get_connection() as conn:
                # Build WHERE clause
                where_clauses = []
                params = {}

                if qa_id:
                    where_clauses.append("qa_id = %(qa_id)s")
                    params["qa_id"] = qa_id

                if pipeline_id_id is not None:
                    where_clauses.append("pipeline_id_id = %(pipeline_id_id)s")
                    params["pipeline_id_id"] = pipeline_id_id

                if qa_type:
                    where_clauses.append("qa_type = %(qa_type)s")
                    params["qa_type"] = qa_type

                if filter_name:
                    where_clauses.append("filter_name = %(filter_name)s")
                    params["filter_name"] = filter_name

                # Build query
                query = """
                    SELECT 
                        id, qa_id, qa_type, imagetyp, filter_name, clipmed, clipstd,
                        clipmin, clipmax, nhotpix, ntotpix, seeing, ellipticity,
                        rotang1, astrometric_offset, skyval, skysig, zp_auto, ezp_auto,
                        ul5_5, stdnumb, created_at, updated_at, pipeline_id_id,
                        qa1, qa2, qa3, qa4, qa5, qa6, qa7, qa8, qa9, qa10
                    FROM pipeline_qadata
                """

                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)

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
            raise PipelineDBError(f"Failed to read QA data: {e}")

    def update_qa_data(self, qa_id: str, **kwargs) -> bool:
        """Update QA data record"""
        try:
            with self.get_connection() as conn:
                if not kwargs:
                    raise PipelineDBError("No fields to update")

                # Build SET clause
                set_clauses = []
                params = {"qa_id": qa_id}

                for key, value in kwargs.items():
                    set_clauses.append(f"{key} = %({key})s")
                    params[key] = value

                # Add updated_at timestamp
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")

                query = f"""
                    UPDATE pipeline_qadata 
                    SET {', '.join(set_clauses)}
                    WHERE qa_id = %(qa_id)s
                """

                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise PipelineDBError(f"No QA record found with ID {qa_id}")

                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to update QA data: {e}")

    def delete_qa_data(self, qa_id: str) -> bool:
        """Delete a QA data record"""
        try:
            with self.get_connection() as conn:
                query = "DELETE FROM pipeline_qadata WHERE qa_id = %s"

                with conn.cursor() as cur:
                    cur.execute(query, (qa_id,))
                    rows_affected = cur.rowcount
                    conn.commit()

                    if rows_affected == 0:
                        raise PipelineDBError(f"No QA record found with ID {qa_id}")

                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to delete QA data: {e}")

    def get_pipeline_with_qa(self, pipeline_id: int) -> Optional[PipelineData]:
        """Get pipeline data with associated QA data"""
        try:
            # Get pipeline data
            pipeline_record = self.read_pipeline_data(pipeline_id)
            if not pipeline_record:
                return None

            # Get associated QA data
            qa_data = self.read_qa_data(pipeline_id_id=pipeline_id)
            pipeline_record.qa_data = qa_data

            return pipeline_record

        except Exception as e:
            raise PipelineDBError(f"Failed to get pipeline with QA data: {e}")

    def find_existing_pipeline_record(
        self, run_date: str, data_type: str, unit: str, config_file: Optional[str] = None
    ) -> Optional[int]:
        """Find existing pipeline record by configuration parameters"""
        try:
            with self.get_connection() as conn:
                where_clauses = []
                params = {"run_date": run_date, "data_type": data_type, "unit": unit}

                # For masterframe data, obj and filt should be NULL
                where_clauses.append("obj IS NULL")
                where_clauses.append("filt IS NULL")

                if config_file:
                    where_clauses.append("config_file = %(config_file)s")
                    params["config_file"] = config_file

                where_clause = " AND ".join(where_clauses)

                query = f"""
                    SELECT id FROM pipeline_pipelinedata
                    WHERE date = %(run_date)s 
                    AND data_type = %(data_type)s 
                    AND unit = %(unit)s
                    AND {where_clause}
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
            raise PipelineDBError(f"Failed to find existing pipeline record: {e}")

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
                        raise PipelineDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to add warning: {e}")

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
                        raise PipelineDBError(f"No pipeline record found with ID {pipeline_id}")

                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to add error: {e}")

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
            pipeline_filename = f"{base_filename}_process.ecsv"
            qa_filename = f"{base_filename}_qa.ecsv"

            # Export pipeline data
            pipeline_data = self._export_pipeline_data_to_ecsv(pipeline_filename)

            # Export QA data
            qa_data = self._export_qa_data_to_ecsv(qa_filename)

            return {"pipeline_data": pipeline_filename, "qa_data": qa_filename}

        except Exception as e:
            raise PipelineDBError(f"Failed to export database to ECSV: {e}")

    def _export_pipeline_data_to_ecsv(self, filename: str) -> bool:
        """Export pipeline data to ECSV file"""
        try:
            with self.get_connection() as conn:
                # Get all pipeline data
                query = """
                    SELECT 
                        id, tag_id, date, data_type, obj, filt, unit, status, progress,
                        bias_exists, dark_filters, flat_filters, warnings, errors, comments,
                        config_file, log_file, debug_file, comments_file, 
                        output_combined_frame_id, created_at, updated_at,
                        param1, param2, param3, param4, param5, param6, param7, param8, param9, param10
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
            raise PipelineDBError(f"Failed to export pipeline data: {e}")

    def _export_qa_data_to_ecsv(self, filename: str) -> bool:
        """Export QA data to ECSV file"""
        try:
            with self.get_connection() as conn:
                # Get all QA data
                query = """
                    SELECT 
                        id, qa_id, qa_type, imagetyp, filter_name, clipmed, clipstd,
                        clipmin, clipmax, nhotpix, ntotpix, seeing, ellipticity,
                        rotang1, astrometric_offset, skyval, skysig, zp_auto, ezp_auto,
                        ul5_5, stdnumb, created_at, updated_at, pipeline_id_id,
                        qa1, qa2, qa3, qa4, qa5, qa6, qa7, qa8, qa9, qa10
                    FROM pipeline_qadata
                    ORDER BY created_at DESC
                """

                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

                    if not rows:
                        print(f"No QA data found to export")
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

                    print(f"Exported {len(rows)} QA records to {filename}")
                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to export QA data: {e}")

    def clear_database(self) -> bool:
        """Clear all data from pipeline tables"""
        try:
            with self.get_connection() as conn:
                # Clear QA data first (due to foreign key constraints)
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM pipeline_qadata")
                    qa_deleted = cur.rowcount

                    # Clear pipeline data
                    cur.execute("DELETE FROM pipeline_pipelinedata")
                    pipeline_deleted = cur.rowcount

                    conn.commit()

                    print(f"Cleared {qa_deleted} QA records and {pipeline_deleted} pipeline records")
                    return True

        except Exception as e:
            raise PipelineDBError(f"Failed to clear database: {e}")


class PipelineDBError(Exception):
    pass
