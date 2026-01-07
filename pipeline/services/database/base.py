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

    def _execute_query(self, query: str, params: Dict[str, Any] = None) -> List[tuple]:
        """Execute a query and return results"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or {})
                return cur.fetchall()

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
