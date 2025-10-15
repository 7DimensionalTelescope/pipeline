import psycopg
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
import json
from contextlib import contextmanager
import os
import pandas as pd

from .const import DB_PARAMS


class DatabaseError(Exception):
    """Base exception for database operations"""

    pass


class BaseDatabase:
    """Base class for database operations with common functionality"""

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
                conn.close()
            raise e
        finally:
            if conn:
                conn.close()

    def _build_where_clause(self, filters: Dict[str, Any]) -> tuple:
        """
        Build WHERE clause and parameters from filter dictionary.

        Args:
            filters: Dictionary of field names to values for filtering

        Returns:
            Tuple of (where_clauses, params) where where_clauses is a list
            of SQL conditions and params is a dictionary of parameters
        """
        where_clauses = []
        params = {}

        for field, value in filters.items():
            if value is not None:
                if field == "run_date" and isinstance(value, str):
                    # Handle date conversion for run_date field (maps to 'date' column)
                    try:
                        from datetime import datetime

                        if len(value) == 10 and value.count("-") == 2:
                            parsed_date = datetime.strptime(value, "%Y-%m-%d").date()
                            where_clauses.append("date = %(run_date)s")
                            params["run_date"] = parsed_date
                        else:
                            where_clauses.append("date::text = %(run_date)s")
                            params["run_date"] = value
                    except ValueError:
                        where_clauses.append("date::text = %(run_date)s")
                        params["run_date"] = value
                else:
                    where_clauses.append(f"{field} = %({field})s")
                    params[field] = value

        return where_clauses, params

    def _execute_query(self, conn, query: str, params: Dict[str, Any] = None) -> List[tuple]:
        """Execute a query and return results"""
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            return cur.fetchall()

    def _execute_update(self, conn, query: str, params: Dict[str, Any] = None) -> int:
        """Execute an update query and return number of affected rows"""
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            rows_affected = cur.rowcount
            conn.commit()
            return rows_affected

    def _execute_insert(self, conn, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute an insert query and return the result"""
        with conn.cursor() as cur:
            cur.execute(query, params or {})
            result = cur.fetchone()
            conn.commit()
            return result

    def _prepare_insert_params(
        self, data_dict: Dict[str, Any], json_fields: List[str] = None, timestamp_fields: List[str] = None
    ) -> tuple:
        """
        Prepare parameters for INSERT query.

        Args:
            data_dict: Dictionary of data to insert
            json_fields: List of field names that should be JSON encoded
            timestamp_fields: List of field names that should use CURRENT_TIMESTAMP

        Returns:
            Tuple of (columns, placeholders, params)
        """
        json_fields = json_fields or []
        timestamp_fields = timestamp_fields or []

        params = data_dict.copy()
        columns = list(params.keys())
        placeholders = [f"%({col})s" for col in columns]

        # Handle JSON fields
        for field in json_fields:
            if field in params:
                params[field] = json.dumps(params[field]) if isinstance(params[field], list) else json.dumps([])

        # Add timestamp fields
        for field in timestamp_fields:
            columns.append(field)
            placeholders.append("CURRENT_TIMESTAMP")

        return columns, placeholders, params

    def _prepare_update_params(self, data_dict: Dict[str, Any], json_fields: List[str] = None) -> tuple:
        """
        Prepare parameters for UPDATE query.

        Args:
            data_dict: Dictionary of data to update
            json_fields: List of field names that should be JSON encoded

        Returns:
            Tuple of (set_clauses, params)
        """
        json_fields = json_fields or []

        set_clauses = []
        params = {}

        for key, value in data_dict.items():
            if key in json_fields:
                params[key] = json.dumps(value) if isinstance(value, list) else json.dumps([])
            else:
                params[key] = value
            set_clauses.append(f"{key} = %({key})s")

        # Add updated_at timestamp
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")

        return set_clauses, params

    def export_to_table(self, table_name: str, order_by: str = "created_at DESC") -> pd.DataFrame:
        """
        Export table data to pandas DataFrame.

        Args:
            table_name: Name of the table to export
            order_by: ORDER BY clause for the query

        Returns:
            pandas DataFrame with the table data
        """
        try:
            with self.get_connection() as conn:
                query = f"SELECT * FROM {table_name} ORDER BY {order_by}"

                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()

                    if not rows:
                        print(f"No data found in {table_name}")
                        return pd.DataFrame()

                    # Get column names
                    columns = [desc[0] for desc in cur.description]

                    # Create pandas DataFrame
                    df = pd.DataFrame(rows, columns=columns)
                    return df
        except Exception as e:
            raise DatabaseError(f"Failed to export {table_name} to table: {e}")

    def export_to_csv(self, table_name: str, filename: str, order_by: str = "created_at DESC") -> bool:
        """
        Export table data to CSV file.

        Args:
            table_name: Name of the table to export
            filename: Output CSV filename
            order_by: ORDER BY clause for the query

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get table data using the table function
            df = self.export_to_table(table_name, order_by)

            if df.empty:
                print(f"No data found in {table_name} to export")
                return False

            # Write to CSV file using pandas
            df.to_csv(filename, index=False)

            print(f"Exported {len(df)} records from {table_name} to {filename}")
            return True

        except Exception as e:
            raise DatabaseError(f"Failed to export {table_name} to CSV: {e}")

    def clear_table(self, table_name: str) -> bool:
        """
        Clear all data from a table.

        Args:
            table_name: Name of the table to clear

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {table_name}")
                    deleted_count = cur.rowcount
                    conn.commit()

                    print(f"Cleared {deleted_count} records from {table_name}")
                    return True

        except Exception as e:
            raise DatabaseError(f"Failed to clear {table_name}: {e}")
