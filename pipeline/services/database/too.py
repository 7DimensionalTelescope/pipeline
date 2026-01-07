import sqlite3
import json
import os
import time
import pytz
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, date
from ...const import TOO_DB_PATH


# Database connection settings
DB_TIMEOUT = 1.0  # Timeout in seconds for database operations
DB_RETRY_MAX_ATTEMPTS = 3  # Maximum number of retry attempts
DB_RETRY_DELAY = 0.1  # Initial retry delay in seconds


class TooDBError(Exception):
    """Exception for ToO database operations"""

    pass


class TooDB:
    """Database class for managing ToO (Target of Opportunity) requests"""

    def __init__(self):
        """Initialize the ToO database connection"""
        self.too_id = None

    def _retry_db_operation(self, operation, *args, **kwargs):
        """
        Retry a database operation with exponential backoff if it encounters a lock.

        Args:
            operation: Callable that performs the database operation
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            TooDBError: If operation fails after all retries
        """
        last_exception = None
        delay = DB_RETRY_DELAY

        for attempt in range(DB_RETRY_MAX_ATTEMPTS):
            try:
                return operation(*args, **kwargs)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                error_msg = str(e).lower()
                # Check if it's a locking error
                if "locked" in error_msg or "database is locked" in error_msg:
                    last_exception = e
                    if attempt < DB_RETRY_MAX_ATTEMPTS - 1:
                        # Exponential backoff with jitter
                        time.sleep(delay * (2**attempt) + (time.time() % 0.1))
                        continue
                # For non-locking errors, raise immediately
                raise TooDBError(f"Database operation failed: {e}") from e
            except Exception as e:
                # For other exceptions, raise immediately
                raise

        # If we exhausted retries, raise the last exception
        raise TooDBError(
            f"Database operation failed after {DB_RETRY_MAX_ATTEMPTS} attempts: {last_exception}"
        ) from last_exception

    def get_connection(self):
        """Create a new database connection with timeout and WAL mode"""
        conn = sqlite3.connect(TOO_DB_PATH, check_same_thread=False, timeout=DB_TIMEOUT)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency (reduces locking)
        # Note: WAL mode is persistent, so this is safe to call every time
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            # WAL might not be supported in some configurations, continue without it
            pass
        # Note: busy_timeout is already set by the timeout parameter in connect()
        return conn

    def _ensure_column_exists(self, column_name: str, column_type: str = "TEXT"):
        """Ensure a column exists in the table, add it if it doesn't"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Check if column exists
            cursor.execute("PRAGMA table_info(too_requests)")
            columns = [row[1] for row in cursor.fetchall()]

            if column_name not in columns:
                try:
                    cursor.execute(f"ALTER TABLE too_requests ADD COLUMN {column_name} {column_type}")
                    conn.commit()
                except sqlite3.OperationalError as e:
                    # Column might have been added by another thread
                    if "duplicate column" not in str(e).lower():
                        raise
        finally:
            conn.close()

    def _convert_row_to_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a dictionary with proper type conversions"""
        if not row:
            return None

        row_dict = dict(row)

        # Convert integer back to boolean for v1 and v2
        if "v1" in row_dict:
            row_dict["v1"] = bool(row_dict["v1"]) if row_dict["v1"] is not None else False
        if "v2" in row_dict:
            row_dict["v2"] = bool(row_dict["v2"]) if row_dict["v2"] is not None else False

        # Convert file_list from JSON string to list
        if "file_list" in row_dict and row_dict["file_list"]:
            try:
                row_dict["file_list"] = json.loads(row_dict["file_list"])
            except (json.JSONDecodeError, TypeError):
                if isinstance(row_dict["file_list"], str):
                    row_dict["file_list"] = [row_dict["file_list"]] if row_dict["file_list"].strip() else []
                else:
                    row_dict["file_list"] = []
        elif "file_list" in row_dict and not row_dict["file_list"]:
            row_dict["file_list"] = []

        # Convert datetime fields from ISO strings to datetime objects
        datetime_fields = ["trigger_time", "observation_time", "transfer_time", "processed_time"]
        for field in datetime_fields:
            if field in row_dict and row_dict[field]:
                try:
                    if isinstance(row_dict[field], str):
                        dt_str = row_dict[field].replace("Z", "").replace("+00:00", "")
                        if "+" in dt_str:
                            dt_str = dt_str.split("+")[0]
                        if "-" in dt_str and dt_str.count("-") > 2:
                            parts = dt_str.rsplit("-", 1)
                            if len(parts) == 2 and ":" in parts[1]:
                                dt_str = parts[0]
                        row_dict[field] = datetime.fromisoformat(dt_str.replace(" ", "T"))
                except (ValueError, AttributeError):
                    pass

        return row_dict

    def _parse_datetime(self, value: Union[datetime, str, None], field_name: str = "datetime") -> Optional[datetime]:
        """
        Parse a datetime value from various formats.

        Args:
            value: datetime object, ISO string, or None
            field_name: Name of the field for error messages

        Returns:
            datetime object or None

        Raises:
            TooDBError: If value cannot be parsed
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            # Clean the string first (remove timezone indicators)
            cleaned = value.replace("Z", "").replace("+00:00", "")
            # Remove timezone offset if present (e.g., "+05:30" or "-08:00")
            if "+" in cleaned and cleaned.count("+") == 1:
                cleaned = cleaned.split("+")[0]
            elif "-" in cleaned and cleaned.count("-") == 3:  # Date has 2 dashes, timezone has 1
                # Split on last dash that's followed by digits (timezone)
                parts = cleaned.rsplit("-", 1)
                if len(parts) == 2 and ":" in parts[1]:
                    cleaned = parts[0]

            # Try strptime formats first, prioritizing formats that match the string structure
            # Check if it has 'T' separator (ISO format)
            if "T" in cleaned:
                # ISO format with T separator
                if "." in cleaned:
                    # Has milliseconds/microseconds
                    formats = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]
                else:
                    # No milliseconds
                    formats = ["%Y-%m-%dT%H:%M:%S"]
            else:
                # Space-separated or date-only formats
                formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"]

            for fmt in formats:
                try:
                    return datetime.strptime(cleaned, fmt)
                except ValueError:
                    continue

            # Fall back to fromisoformat (handles ISO with T, milliseconds, timezone, etc.)
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                pass

            raise TooDBError(f"Failed to parse {field_name}: {value}")

        raise TooDBError(f"Invalid {field_name} type: {type(value)}")

    def _prepare_update_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for update with proper conversions"""
        clean_data = {}

        for key, value in data.items():
            if key in ["id", "created_at", "updated_at"]:
                continue
            if isinstance(value, dict):
                # Convert dict to JSON string
                clean_data[key] = json.dumps(value)
            elif isinstance(value, list):
                # Special handling for file_list - convert to JSON string
                if key == "file_list":
                    clean_data[key] = json.dumps(value) if value else None
                else:
                    clean_data[key] = value[0] if len(value) > 0 else None
            else:
                clean_data[key] = value

        # Convert boolean to integer for v1 and v2 (SQLite stores as INTEGER)
        if "v1" in clean_data:
            if isinstance(clean_data["v1"], bool):
                clean_data["v1"] = 1 if clean_data["v1"] else 0
            elif clean_data["v1"] is None:
                clean_data["v1"] = 0
            else:
                clean_data["v1"] = 1 if clean_data["v1"] else 0
        if "v2" in clean_data:
            if isinstance(clean_data["v2"], bool):
                clean_data["v2"] = 1 if clean_data["v2"] else 0
            elif clean_data["v2"] is None:
                clean_data["v2"] = 0
            else:
                clean_data["v2"] = 1 if clean_data["v2"] else 0

        # Convert v2_progress to float
        if "v2_progress" in clean_data and clean_data["v2_progress"] is not None:
            try:
                clean_data["v2_progress"] = float(clean_data["v2_progress"])
            except (ValueError, TypeError):
                clean_data["v2_progress"] = None

        # Convert file_list to JSON string if it's a list (fallback if not handled above)
        if "file_list" in clean_data and clean_data["file_list"] is not None:
            if isinstance(clean_data["file_list"], list):
                clean_data["file_list"] = json.dumps(clean_data["file_list"])
            elif not isinstance(clean_data["file_list"], str):
                clean_data["file_list"] = json.dumps([clean_data["file_list"]])

        # Convert datetime fields to ISO format strings
        datetime_fields = ["trigger_time", "observation_time", "transfer_time", "processed_time"]
        for field in datetime_fields:
            if field in clean_data and clean_data[field] is not None:
                if isinstance(clean_data[field], datetime):
                    clean_data[field] = clean_data[field].isoformat()
                elif not isinstance(clean_data[field], str):
                    clean_data[field] = str(clean_data[field])

        return clean_data

    def _parse_config_filename(self, config_file: str) -> Dict[str, Optional[str]]:
        """
        Parse config filename to extract tile, objname, date, and timestamp.

        Format: T19154_m675_2025-11-10_too_022205.yml
        - Date is before "too"
        - Timestamp is after "too"
        - First part can be tile (T followed by numbers) or object name

        Args:
            config_file: Config file path or filename

        Returns:
            Dictionary with tile, objname, date, and timestamp
        """
        # Extract just the filename if full path is provided
        filename = os.path.basename(config_file)

        # Remove .yml extension
        base_name = filename.replace(".yml", "").replace(".yaml", "")

        parts = base_name.split("_")
        result = {
            "tile": None,
            "objname": None,
            "date": None,
            "timestamp": None,
        }

        # Find "too" marker to locate the date and timestamp
        too_index = None
        for i, part in enumerate(parts):
            if part.lower() == "too":
                too_index = i
                break

        if too_index is not None and too_index >= 1 and too_index + 1 < len(parts):
            # Date is after "too", timestamp is after "too"

            potential_date = parts[too_index + 1]
            timestamp = parts[too_index + 2]

            # Validate date format (YYYY-MM-DD)
            if len(potential_date) == 8:
                result["date"] = f"{potential_date[0:4]}-{potential_date[4:6]}-{potential_date[6:8]}"

            # Store timestamp as-is (HHMMSS format)
            if timestamp.isdigit() and len(timestamp) >= 6:
                result["timestamp"] = timestamp

        # Check if first part is a tile (starts with 'T' followed by numbers)
        if parts and parts[0].startswith("T") and parts[0][1:].isdigit():
            result["tile"] = parts[0]
        elif parts and len(parts[0]) > 0:
            # Otherwise, first part is object name
            result["objname"] = parts[0]

        return result

    def _convert_chile_time_to_utc(self, date_str: str, time_str: str) -> str:
        """
        Convert Chile time (from config filename) to UTC for database comparison.

        Args:
            date_str: Date string in YYYY-MM-DD format
            time_str: Time string in HH:MM:SS format

        Returns:
            UTC datetime as string in YYYY-MM-DD HH:MM:SS format
        """
        chile_tz = pytz.timezone("America/Santiago")  # Chile timezone (handles DST automatically)
        obs_time_chile_str = f"{date_str} {time_str}"

        # Parse as naive datetime, localize to Chile time, then convert to UTC
        obs_time_chile = datetime.strptime(obs_time_chile_str, "%Y-%m-%d %H:%M:%S")
        obs_time_chile = chile_tz.localize(obs_time_chile)
        obs_time_utc = obs_time_chile.astimezone(pytz.UTC)

        # Convert to string format for database comparison
        return obs_time_utc.strftime("%Y-%m-%d %H:%M:%S")

    def _query_too_by_date_and_time(
        self, cursor, identifier: str, identifier_value: str, observation_date: str, observation_time_str: str
    ) -> Optional[Dict[str, Any]]:
        """
        Query ToO records by identifier (tile or objname) with date and time constraints.

        Args:
            cursor: Database cursor
            identifier: Either 'tile' or 'objname'
            identifier_value: Value of the identifier
            observation_date: Date string for matching
            observation_time_str: UTC time string for comparison

        Returns:
            Dictionary with ToO data if found, None otherwise
        """
        # First try: same date as observation_time, return most recent trigger_time on that date
        cursor.execute(
            f"""SELECT * FROM too_requests 
               WHERE {identifier} = ? AND DATE(trigger_time) = ? 
               ORDER BY trigger_time DESC, created_at DESC 
               LIMIT 1""",
            (identifier_value, observation_date),
        )
        row = cursor.fetchone()
        if row:
            return self._convert_row_to_dict(row)

        # Fallback: any date, trigger_time <= observation_time
        cursor.execute(
            f"""SELECT * FROM too_requests 
               WHERE {identifier} = ? AND trigger_time <= ? 
               ORDER BY trigger_time DESC, created_at DESC 
               LIMIT 1""",
            (identifier_value, observation_time_str),
        )
        row = cursor.fetchone()
        if row:
            return self._convert_row_to_dict(row)

        return None

    # ==================== TOO DATA MANAGEMENT ====================

    def read_too_data_by_id(self, too_id: int) -> Optional[Dict[str, Any]]:
        """
        Read ToO data record by ID.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM too_requests WHERE id = ?", (too_id,))
            row = cursor.fetchone()
            if row:
                return self._convert_row_to_dict(row)
            return None
        finally:
            conn.close()

    def read_too_data(
        self,
        config_file: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Read ToO data record from config file.

        Args:
            config_file: Config file path (e.g., "T02252_i_2025-11-19_too_2025-11-19_123123.yml" or
                         "AT2024hfx_i_2025-11-19_too_2025-11-19_123123.yml")
                         Will parse filename to extract tile/objname, date, and timestamp, then query database
                         for records where trigger_time <= observation_time (from parsed date/timestamp)

        Returns:
            Dictionary with ToO data if found, None otherwise
        """
        if not config_file:
            raise TooDBError("config_file is required")

        # Parse config filename to extract tile, date, and timestamp
        parsed = self._parse_config_filename(config_file)

        if not parsed["date"] or not parsed["timestamp"]:
            raise TooDBError(f"Could not parse date and timestamp from config file: {config_file}")

        # Convert timestamp from HHMMSS format to HH:MM:SS
        timestamp = parsed["timestamp"]
        if len(timestamp) < 6:
            raise TooDBError(f"Invalid timestamp format: {timestamp}")
        time_str = f"{timestamp[0:2]}:{timestamp[2:4]}:{timestamp[4:6]}"

        # Convert Chile time (from config filename) to UTC for database comparison
        observation_time_str = self._convert_chile_time_to_utc(parsed["date"], time_str)
        observation_date = parsed["date"]

        # Query database: find records where trigger_time <= observation_time
        # Prioritize records from the same date as observation_time
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Try most specific match first: tile/objname + same date as observation_time
            if parsed["tile"]:
                row_dict = self._query_too_by_date_and_time(
                    cursor, "tile", parsed["tile"], observation_date, observation_time_str
                )
                if row_dict:
                    self.too_id = row_dict.get("id")
                    return row_dict

            if parsed["objname"]:
                row_dict = self._query_too_by_date_and_time(
                    cursor, "objname", parsed["objname"], observation_date, observation_time_str
                )
                if row_dict:
                    self.too_id = row_dict.get("id")
                    return row_dict

            # If no match with tile/objname, try with trigger_time <= observation_time alone
            cursor.execute(
                """SELECT * FROM too_requests 
                   WHERE trigger_time <= ? 
                   ORDER BY trigger_time DESC, created_at DESC 
                   LIMIT 1""",
                (observation_time_str,),
            )
            row = cursor.fetchone()
            if row:
                row_dict = self._convert_row_to_dict(row)
                self.too_id = row_dict.get("id")
                return row_dict

            self.too_id = None
            return None
        finally:
            conn.close()

    def update_too_data(
        self,
        too_id: Optional[int] = None,
        tile: Optional[str] = None,
        objname: Optional[str] = None,
        trigger_time: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """
        Update ToO data record.

        Args:
            too_id: ToO request ID (optional, will use self.too_id if not provided)
            tile: Tile name (e.g., "T13201") - used only if too_id and self.too_id are None
            objname: Object name (e.g., "AT2024hfx") - used only if too_id and self.too_id are None
            trigger_time: Trigger time (ISO format string) - used only if too_id and self.too_id are None
            **kwargs: Fields to update. Supported fields:
                - observation_time: datetime or ISO string (can also use obs_date)
                - obs_date: Observation date (will be converted to observation_time)
                - processed_time: datetime or ISO string (defaults to current time if not provided)
                - v2: bool (whether process is done)
                - v2_status: str (status like "astrometry", "photometry", "completed")
                - v2_progress: float (0-100, automatically clamped)
                - file_list: List[str] (list of image file paths)

        Returns:
            True if successful, False otherwise

        Raises:
            TooDBError: If too_id cannot be found or update fails
        """
        try:
            if not kwargs:
                raise TooDBError("No fields to update")

            # Find row ID - use parameter first, then self.too_id, then try to find by identifiers
            if too_id is None:
                too_id = self.too_id
            if too_id is None:
                too_id = self._find_row_id(tile=tile, objname=objname, reference_time=trigger_time)
            if too_id is None:
                raise TooDBError(
                    "Could not find ToO record. Use read_too_data() first or provide too_id/tile/objname/trigger_time"
                )

            # Prepare data dictionary
            data = {}

            if "observation_time" in kwargs:
                obs_time = self._parse_datetime(kwargs.pop("observation_time"), "observation_time")
                if obs_time:
                    data["observation_time"] = obs_time.isoformat()

            if "transfer_time" in kwargs:
                transfer_time = self._parse_datetime(kwargs.pop("transfer_time"), "transfer_time")
                if transfer_time:
                    data["transfer_time"] = transfer_time.isoformat()

            # Handle processed_time (default to current time if not provided)
            if "processed_time" in kwargs:
                processed_time = self._parse_datetime(kwargs.pop("processed_time"), "processed_time")
                if processed_time is None:
                    processed_time = datetime.now()
                data["processed_time"] = processed_time.isoformat()
            else:
                # If updating other fields, also update processed_time to current time
                data["processed_time"] = datetime.now().isoformat()

            # Handle v2_progress (clamp to 0-100)
            if "v2_progress" in kwargs:
                v2_progress = kwargs.pop("v2_progress")
                data["v2_progress"] = max(0.0, min(100.0, float(v2_progress)))

            # Copy remaining kwargs to data
            for key, value in kwargs.items():
                data[key] = value

            if not data:
                raise TooDBError("No valid fields to update")

            # Prepare data for update
            clean_data = self._prepare_update_data(data)

            # Ensure all columns exist
            for key in clean_data.keys():
                if key not in ["id", "created_at", "updated_at"]:
                    self._ensure_column_exists(key)

            # Update the row
            conn = self.get_connection()
            try:
                cursor = conn.cursor()

                set_clause = ", ".join([f"{key} = ?" for key in clean_data.keys()])
                set_clause += ", updated_at = CURRENT_TIMESTAMP"
                values = list(clean_data.values()) + [too_id]

                cursor.execute(f"UPDATE too_requests SET {set_clause} WHERE id = ?", values)
                conn.commit()

                if cursor.rowcount == 0:
                    raise TooDBError(f"Failed to update ToO record with ID {too_id}")

                return True
            finally:
                conn.close()

        except TooDBError:
            raise
        except Exception as e:
            raise TooDBError(f"Failed to update ToO data: {e}")

    def update_too_progress(self, too_id: int, progress: int, status: Optional[str] = None) -> bool:
        """
        Update pipeline progress in database.
        This method is used by the handler to update progress and status.

        Args:
            too_id: ToO request ID
            progress: Progress percentage (0-100)
            status: Optional status string (e.g., "astrometry", "photometry", "completed")

        Returns:
            True if successful, False otherwise

        Raises:
            TooDBError: If update fails
        """
        try:
            # Clamp progress to 0-100
            progress = max(0, min(100, int(progress)))

            # Use update_too_data to update progress and status
            update_data = {
                "v2_progress": float(progress),
            }

            # Add status if provided
            if status:
                update_data["v2_status"] = status

            if float(progress) >= 100:
                update_data["v2"] = True

            self.check_v1_images(too_id=too_id)

            # Update the row using update_too_data
            return self.update_too_data(too_id=too_id, **update_data)

        except TooDBError:
            raise
        except Exception as e:
            raise TooDBError(f"Failed to update pipeline progress: {e}")

    def _find_row_id(
        self,
        tile: Optional[str] = None,
        objname: Optional[str] = None,
        reference_time: Optional[str] = None,
    ) -> Optional[int]:
        """
        Find ToO record ID by identifiers.
        Priority: trigger_time + (tile or objname) > trigger_time > tile > objname

        Args:
            tile: Tile name (e.g., "T13201")
            objname: Object name (e.g., "AT2024hfx")
            reference_time: Trigger time (ISO format string) - will find nearest past match

        Returns:
            Row ID if found, None otherwise
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()

            # Try most specific match first: trigger_time + (tile or objname)
            if reference_time:
                if tile:
                    cursor.execute(
                        """SELECT id FROM too_requests 
                           WHERE tile = ? AND trigger_time <= ? 
                           ORDER BY trigger_time DESC, created_at DESC 
                           LIMIT 1""",
                        (tile, reference_time),
                    )
                    result = cursor.fetchone()
                    if result:
                        return result[0]

                if objname:
                    cursor.execute(
                        """SELECT id FROM too_requests 
                           WHERE objname = ? AND trigger_time <= ? 
                           ORDER BY trigger_time DESC, created_at DESC 
                           LIMIT 1""",
                        (objname, reference_time),
                    )
                    result = cursor.fetchone()
                    if result:
                        return result[0]

            # Try objname alone
            if objname:
                cursor.execute(
                    "SELECT id FROM too_requests WHERE objname = ? ORDER BY created_at DESC LIMIT 1", (objname,)
                )
                result = cursor.fetchone()
                if result:
                    return result[0]

            # Try tile alone
            if tile:
                cursor.execute("SELECT id FROM too_requests WHERE tile = ? ORDER BY created_at DESC LIMIT 1", (tile,))
                result = cursor.fetchone()
                if result:
                    return result[0]

            # Try trigger_time alone - find nearest past row
            if reference_time:
                cursor.execute(
                    """SELECT id FROM too_requests 
                       WHERE trigger_time <= ? 
                       ORDER BY trigger_time DESC, created_at DESC 
                       LIMIT 1""",
                    (reference_time,),
                )
                result = cursor.fetchone()
                if result:
                    return result[0]

            return None
        finally:
            conn.close()

    def check_v1_images(self, too_id: Optional[int] = None) -> bool:
        """
        Check if calibrated science images exist for a ToO and update v1 flag in database.

        Args:
            too_id: ToO request ID (optional, will use self.too_id if not provided)

        Returns:
            True if calibrated science images exist, False otherwise

        Raises:
            TooDBError: If too_id cannot be found or query fails
        """
        try:
            # Get ToO data
            if too_id is None:
                too_id = self.too_id
            if too_id is None:
                raise TooDBError("too_id is required. Use read_too_data() first or provide too_id")

            too_data = self.read_too_data_by_id(too_id)
            if not too_data:
                raise TooDBError(f"ToO record with ID {too_id} not found")

            # Extract observation date
            observation_time = too_data.get("observation_time")
            if not observation_time:
                raise TooDBError("observation_time not found in ToO data")

            # Convert to date object
            if isinstance(observation_time, datetime):
                obs_date = observation_time.date()
            elif isinstance(observation_time, str):
                # Parse ISO format string
                try:
                    obs_date = datetime.fromisoformat(observation_time.replace("Z", "")).date()
                except ValueError:
                    # Try other formats
                    obs_date = datetime.strptime(observation_time.split()[0], "%Y-%m-%d").date()
            else:
                raise TooDBError(f"Invalid observation_time format: {type(observation_time)}")

            # Query PostgreSQL database for calibrated science images
            try:
                from .query import get_pool

                pool = get_pool()
                if pool is None:
                    raise TooDBError("PostgreSQL connection pool not available")

                has_calibrated = False
                with pool.connection() as conn:
                    with conn.cursor() as cur:
                        # Check for science frames on the observation date (regardless of unit)
                        science_query = """
                            SELECT EXISTS(
                                SELECT 1 
                                FROM survey_scienceframe sf
                                JOIN survey_night n ON sf.night_id = n.id
                                WHERE n.date = %s
                                LIMIT 1
                            )
                        """
                        cur.execute(science_query, (obs_date,))
                        has_calibrated = cur.fetchone()[0]

                # Update v1 flag in too_requests table
                self.update_too_data(too_id=too_id, v1=has_calibrated)

                return has_calibrated

            except ImportError:
                raise TooDBError("Failed to import PostgreSQL query modules")
            except Exception as e:
                raise TooDBError(f"Failed to query calibrated science images: {e}")

        except TooDBError:
            raise
        except Exception as e:
            raise TooDBError(f"Failed to check calibrated science images: {e}")

    # ==================== EMAIL NOTIFICATION HELPERS ====================

    def _format_datetime(self, dt):
        """Format datetime object or string to readable format in KST."""
        from datetime import datetime

        if dt is None:
            return "N/A"
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace("Z", ""))
            except (ValueError, AttributeError):
                return dt
        if isinstance(dt, datetime):
            # Database stores datetimes in KST, so naive datetimes are already in KST
            if dt.tzinfo is None:
                # Naive datetime - already in KST, just format it
                return dt.strftime("%Y-%m-%d %H:%M:%S KST")
            else:
                # Timezone-aware datetime - convert to KST
                # Note: astimezone is safe to call even if already in KST (from sciprocess.py)
                kst = pytz.timezone("Asia/Seoul")
                dt_kst = dt.astimezone(kst)
                return dt_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        return str(dt)

    def _format_coord(self, coord, coord_deg):
        """Format coordinate with degree value if available."""
        if coord_deg is not None:
            return f"{coord_deg:.6f}°"
        elif coord:
            return coord
        return "N/A"

    def _format_timedelta(self, start_time, end_time):
        """Calculate and format time difference between two datetimes."""
        from datetime import datetime

        if start_time is None or end_time is None:
            return "N/A"

        # Convert to datetime objects if needed
        def to_datetime(dt):
            if isinstance(dt, datetime):
                return dt
            if isinstance(dt, str):
                try:
                    return datetime.fromisoformat(dt.replace("Z", ""))
                except (ValueError, AttributeError):
                    return None
            return None

        start = to_datetime(start_time)
        end = to_datetime(end_time)

        if start is None or end is None:
            return "N/A"

        # Calculate difference
        delta = end - start
        total_seconds = int(delta.total_seconds())

        if total_seconds < 0:
            return "N/A"

        # Format as days, hours, minutes
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if not parts and seconds > 0:
            parts.append(f"{seconds}s")

        return " ".join(parts) if parts else "0s"

    def _parse_cc_recipients(self, default_recipient):
        """Parse CC recipients from DEFAULT_RECIPIENT (JSON string, list, or comma-separated string)."""
        import json

        if not default_recipient:
            return None

        try:
            if isinstance(default_recipient, str):
                # Try parsing as JSON first
                try:
                    return json.loads(default_recipient)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, treat as comma-separated string
                    return [email.strip() for email in default_recipient.split(",") if email.strip()]
            elif isinstance(default_recipient, list):
                return default_recipient
            else:
                return None
        except Exception:
            return None

    def _send_email(self, to, subject, contents, attachments=None, cc=None):
        """Send email using yagmail with proper error handling."""
        import yagmail
        from ...const import EMAIL_PASSWORD, EMAIL_USER

        try:
            user = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASSWORD)

            # Prepare email parameters
            email_kwargs = {"to": to, "subject": subject, "contents": contents}
            if attachments:
                email_kwargs["attachments"] = attachments
            if cc:
                email_kwargs["cc"] = cc

            user.send(**email_kwargs)
            return True
        except Exception as e:
            raise TooDBError(f"Failed to send email: {e}")

    def _find_output_files(
        self, base_path, image_filename="combined_sed_cutouts.png", text_filename="combined_sed_cutouts_output.txt"
    ):
        """Find output files in base_path or subdirectories."""
        from pathlib import Path

        if not base_path:
            return None, None

        base_path_obj = Path(base_path)
        if not base_path_obj.exists():
            return None, None

        # Try direct path first
        image_path = base_path_obj / image_filename
        text_path = base_path_obj / text_filename

        # Search in subdirectories if not found
        if not image_path.exists():
            candidates = list(base_path_obj.rglob(image_filename))
            image_path = candidates[0] if candidates else None

        if not text_path.exists():
            candidates = list(base_path_obj.rglob(text_filename))
            text_path = candidates[0] if candidates else None

        return image_path, text_path

    def _build_subject(self, objname, tile=None, suffix=""):
        """Build email subject line."""
        subject = f"[7DT ToO Alert] {objname}"
        if tile:
            subject += f" (Tile: {tile})"
        if suffix:
            subject += f" - {suffix}"
        return subject

    # ==================== EMAIL NOTIFICATION METHODS ====================

    def send_initial_notice_email(self, too_id: int, test=False) -> bool:
        """
        Send initial email notification when ToO images are transferred.
        Includes observation parameters and timing information.
        """
        from ...const import DEFAULT_RECIPIENT

        too_data = self.read_too_data_by_id(too_id)

        if too_data.get("init_notice") == 1:
            return True
        else:
            self.update_too_data(too_id=too_id, init_notice=1)

        # Extract information from TooDB
        objname = too_data.get("objname", "Unknown")
        tile = too_data.get("tile", "")
        requester = too_data.get("requester", "")

        # Build subject
        subject = self._build_subject(objname, tile, "Images transferred to Proton server")

        # Build contents
        contents = self._build_initial_email_contents(too_data)

        # Handle test mode
        if test:
            requester = "takdg123@gmail.com"
            cc_recipients = ["takdg123@gmail.com"]
        else:
            cc_recipients = self._parse_cc_recipients(DEFAULT_RECIPIENT)

        # Send email
        self._send_email(to=requester, subject=subject, contents=contents, cc=cc_recipients)

        return True

    def _build_too_request_info(self, too_data, include_processed_time=False):
        """Build ToO Request Information section (common for both initial and final notices)."""
        objname = too_data.get("objname", "Unknown")
        tile = too_data.get("tile", "")
        ra = too_data.get("ra", "")
        ra_deg = too_data.get("ra_deg")
        dec = too_data.get("dec", "")
        dec_deg = too_data.get("dec_deg")
        exptime = too_data.get("exptime")
        n_image = too_data.get("n_image")
        gain = too_data.get("gain")
        binning = too_data.get("binning")
        observation_time = too_data.get("observation_time")
        trigger_time = too_data.get("trigger_time")
        transfer_time = too_data.get("transfer_time")
        comments = too_data.get("comments", "")
        processed_time = too_data.get("processed_time") if include_processed_time else None

        contents = f"""
ToO Request Information:
=======================
Object Name: {objname}
"""
        if tile:
            contents += f"Tile: {tile}\n"

        ra_formatted = self._format_coord(ra, ra_deg)
        dec_formatted = self._format_coord(dec, dec_deg)
        ra_suffix = f" ({ra})" if ra and ra_deg else ""
        dec_suffix = f" ({dec})" if dec and dec_deg else ""

        contents += f"""
Coordinates:
  RA: {ra_formatted}{ra_suffix}
  Dec: {dec_formatted}{dec_suffix}

Observation Parameters:
  Exposure Time: {exptime if exptime is not None else 'N/A'} seconds
  Number of Images: {n_image if n_image is not None else 'N/A'}
  Gain: {gain if gain is not None else 'N/A'}
  Binning: {binning if binning is not None else 'N/A'}

Timing Information:
  Trigger Time: {self._format_datetime(trigger_time)}
  Observation Time: {self._format_datetime(observation_time)} (ΔT: {self._format_timedelta(trigger_time, observation_time)})
  Transfer Time: {self._format_datetime(transfer_time)} (ΔT: {self._format_timedelta(observation_time, transfer_time)})
"""
        if include_processed_time and processed_time:
            contents += f"  Completed Processing Time: {self._format_datetime(processed_time)} (ΔT: {self._format_timedelta(observation_time, processed_time)})\n"

        if comments:
            contents += f"\nComments: {comments}\n"

        return contents

    def _build_initial_email_contents(self, too_data):
        """Build email contents for initial notice."""
        contents = self._build_too_request_info(too_data, include_processed_time=False)

        contents += "\nThe images will be automatically processed through the pipeline, and you will receive a notification once processing is complete.\n"
        contents += "\nIf you want to know the status of the processing, you can visit the ToO webpage: https://proton.snu.ac.kr/too/\n"

        return contents

    def send_final_notice_email(self, too_id: int, sed_data=None, test=False, force_to_send=False) -> bool:
        """
        Send final email notification when ToO processing is complete.
        Includes SED plot and magnitude data as attachments.
        """
        from ...const import DEFAULT_RECIPIENT

        self.update_too_data(too_id=too_id, final_notice=1)

        too_data = self.read_too_data_by_id(too_id)

        completed = too_data.get("completed")

        if (completed != too_data.get("num_filters")) and not (force_to_send):
            return True

        # Validate processing is complete
        v2 = too_data.get("v2", False)
        v2_progress = too_data.get("v2_progress", 0)
        if v2_progress < 100:
            return True

        # Extract information
        objname = too_data.get("objname", "Unknown")
        tile = too_data.get("tile", "")
        base_path = too_data.get("base_path", "")
        requester = too_data.get("requester", "")

        # Find output files
        image_path, _ = self._find_output_files(base_path)

        # Build subject and contents
        # sed_data should be passed as parameter (e.g., from make_too_output)
        subject = self._build_subject(objname, tile, "Processing Complete")
        contents = self._build_final_email_contents(too_data, image_path, sed_data, base_path)

        # Prepare attachments
        attachments = []
        if image_path and image_path.exists():
            attachments.append(str(image_path))

        # Handle test mode
        if test:
            requester = "takdg123@gmail.com"
            cc_recipients = ["takdg123@gmail.com"]
        else:
            cc_recipients = self._parse_cc_recipients(DEFAULT_RECIPIENT)

        # Send email
        self._send_email(
            to=requester,
            subject=subject,
            contents=contents,
            attachments=attachments if attachments else None,
            cc=cc_recipients,
        )

        return True

    def _build_final_email_contents(self, too_data, image_path, sed_data, base_path):
        """Build email contents for final notice."""

        contents = """
Final Detection Result:
=======================
The data processing for the ToO observation is complete. The results are as follows:
"""
        # Add SED table if sed_data is available
        if sed_data:
            contents += "\n"
            contents += self._format_sed_table(sed_data)

        if base_path:
            contents += "\n"
            contents += "\nAll output images and catalogs are available in the output directory.\n"
            contents += f"  Output Directory: {base_path}\n"

        if image_path and image_path.exists():
            contents += "\n"
            contents += "\nThe attached file contains the SED plot with magnitude measurements.\n"
            contents += f"  SED Plot: {image_path}\n"

        contents += self._build_too_request_info(too_data, include_processed_time=True)

        return contents

    def _format_sed_table(self, sed_data):
        """Format SED data as a table similar to plotting.py output."""
        from ...too.table import format_sed_table_string

        return format_sed_table_string(sed_data, title="Magnitude Measurements")

    def send_interim_notice_email(self, too_id: int, sed_data=None, dtype="difference", test=False) -> bool:
        """
        Send interim email notification when one filterset is completed.
        Shows detection status, filter used, and magnitude if detected.

        Args:
            too_id: ToO request ID
            sed_data: SED data containing filter and magnitude information
            dtype: Detection type - "difference" for difference image, "stacked" for stacked image
            test: If True, send to test email address

        Returns:
            True if successful

        Raises:
            TooDBError: If too_id cannot be found or email fails
        """
        from ...const import DEFAULT_RECIPIENT

        too_data = self.read_too_data_by_id(too_id)
        
        if too_data.get("num_filters") == 0:
            num_filters = self._count_num_filters(too_id)
        else:
            num_filters = too_data.get("num_filters")
            

        if too_data.get("interim_notice") == 1:
            return True
        else:
            self.update_too_data(too_id=too_id, interim_notice=1)

        # Extract information
        objname = too_data.get("objname", "Unknown")
        tile = too_data.get("tile", "")
        requester = too_data.get("requester", "")

        # Determine detection status from sed_data
        # sed_data is a list of dicts from plot_cutouts_and_sed, each with:
        # filter_name, magnitude, mag_error, is_upper_limit, etc.
        is_detected = False
        filt = None
        mag = None
        mag_error = None
        is_upper_limit = False

        if sed_data:
            # sed_data is a list, get the first item (or find the relevant filter)
            if isinstance(sed_data, list) and len(sed_data) > 0:
                # Get first item (for single filter completion)
                item = sed_data[0] if isinstance(sed_data[0], dict) else sed_data[0]
                filt = item.get("filter_name") if isinstance(item, dict) else getattr(item, "filter_name", None)
                mag = item.get("magnitude") if isinstance(item, dict) else getattr(item, "magnitude", None)
                mag_error = item.get("mag_error") if isinstance(item, dict) else getattr(item, "mag_error", None)
                is_upper_limit = (
                    item.get("is_upper_limit", False)
                    if isinstance(item, dict)
                    else getattr(item, "is_upper_limit", False)
                )
            elif isinstance(sed_data, dict):
                # Handle case where it's a single dict
                filt = sed_data.get("filter_name")
                mag = sed_data.get("magnitude")
                mag_error = sed_data.get("mag_error")
                is_upper_limit = sed_data.get("is_upper_limit", False)

            # Check if detected: magnitude is not NaN/inf, and NOT an upper limit
            # Upper limits mean the source was NOT detected, just that we can set an upper bound
            has_valid_mag = mag is not None and not (isinstance(mag, float) and (mag != mag or mag == float("inf")))
            is_detected = has_valid_mag and not is_upper_limit

        detection_status = "Detected" if is_detected else "Not Detected"

        # Build subject
        filter_str = filt if filt else "Unknown Filter"
        subject = self._build_subject(objname, tile, f"Interim Notice - {filter_str} ({detection_status})")

        # Build contents
        contents = self._build_interim_notice_contents(
            too_data,
            sed_data,
            dtype=dtype,
            is_detected=is_detected,
            mag=mag,
            mag_error=mag_error,
            is_upper_limit=is_upper_limit,
            num_filters=num_filters,
        )

        # Handle test mode
        if test:
            requester = "takdg123@gmail.com"
            cc_recipients = ["takdg123@gmail.com"]
        else:
            cc_recipients = self._parse_cc_recipients(DEFAULT_RECIPIENT)

        # Send email
        self._send_email(to=requester, subject=subject, contents=contents, cc=cc_recipients)

        self.update_too_data(too_id=too_id, interim_notice=1)

        return True

    def _build_interim_notice_contents(
        self, too_data, sed_data, dtype="difference", is_detected=None, mag=None, mag_error=None, is_upper_limit=False, num_filters=None
    ):
        """Build email contents for interim notice."""
        contents = """
Interim Processing Result:
=========================
"""
        contents += "This is an interim notification for a single filter completion.\n\n"

        # Extract filter from sed_data (sed_data is a list of dicts from plot_cutouts_and_sed)
        filt = None
        if sed_data:
            if isinstance(sed_data, list) and len(sed_data) > 0:
                item = sed_data[0] if isinstance(sed_data[0], dict) else sed_data[0]
                filt = item.get("filter_name") if isinstance(item, dict) else getattr(item, "filter_name", None)
            elif isinstance(sed_data, dict):
                filt = sed_data.get("filter_name")

        if is_detected and dtype == "difference":
            contents += f"Detection: Detected\n"
            contents += "Detection Method: Difference Image Analysis\n"
        elif is_detected and dtype == "stacked":
            contents += f"Detection: Possibly Detected. \n"
            contents += "Detection Method: Stacked Image Analysis\n"
        else:
            contents += f"Detection: Not Detected\n"

        filter_str = filt if filt else "Unknown Filter"
        contents += f"Filter: {filter_str}\n"

        if is_detected and mag is not None:
            if is_upper_limit:
                contents += f"Magnitude: >{mag:.3f} (3σ upper limit)\n"
            else:
                if mag_error is not None and not np.isnan(mag_error) and mag_error > 0:
                    contents += f"Magnitude: {mag:.3f} ± {mag_error:.3f}\n"
                else:
                    contents += f"Magnitude: {mag:.3f}\n"
        else:
            contents += "Magnitude: N/A (Not detected)\n"

        contents += "\n"
        if is_detected:
            if dtype == "difference":
                contents += "The target has been detected in the difference image, which indicates a high-confidence "
                contents += "detection. Continued follow-up observations are strongly recommended to confirm and "
                contents += "characterize the transient event.\n"
            elif dtype == "stacked":
                contents += "The target has been detected in the stacked image. Please note that the detected source "
                contents += "may be a pre-existing source or a known object in the field. Cross-matching with "
                contents += "astronomical catalogs (e.g., Gaia, SDSS, Pan-STARRS), visual inspection of the "
                contents += "stacked image, and/or difference image analysis are required to confirm whether this is a new transient event. \n"
        else:
            contents += "The target was not detected in this filter. This may indicate the transient is below "
            contents += "the detection threshold, outside the field of view, or has faded.\n"

        contents += "\n"
        contents += f"Further data processing is ongoing for {num_filters} filters.\n"

        contents += "\nA final notification will be sent once processing for all filters is complete.\n\n"

        contents += self._build_too_request_info(too_data, include_processed_time=False)

        return contents

    def _count_num_filters(self, too_id: int) -> int:
        """Count the number of filters for a given ToO request."""
        too_data = self.read_too_data_by_id(too_id)
        base_path = too_data.get("base_path")

        if not base_path or not os.path.exists(base_path):
            self.update_too_data(too_id=too_id, num_filters=0)
            return 0

        # Count only directories that look like filter names
        # Filter names are: u, g, r, i, z (broadband) or mXXX (medium band)
        from ...const import ALL_FILTERS

        num_filters = 0
        try:
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                # Only count directories (not files) and exclude hidden files
                if os.path.isdir(item_path) and not item.startswith("."):
                    # Check if it's a valid filter name
                    if item in ALL_FILTERS:
                        num_filters += 1
        except (OSError, PermissionError) as e:
            # Log error but don't raise - return 0 instead
            # Note: TooDB doesn't have a logger, so we'll just handle silently
            # or could use print for debugging
            num_filters = 0

        self.update_too_data(too_id=too_id, num_filters=num_filters)
        return num_filters

    def mark_completed(self, too_id: int) -> bool:
        """Mark a ToO request as completed."""

        too_data = self.read_too_data_by_id(too_id)

        completed = too_data.get("completed")

        completed += 1
        self.update_too_data(too_id=too_id, completed=completed)
        return True
