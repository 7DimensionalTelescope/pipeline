import psycopg
from typing import List, Dict, Optional, Any, Union, Tuple
from contextlib import contextmanager
import pandas as pd
from abc import abstractmethod
from .const import DB_PARAMS
from .query_string import *
import json


class DatabaseError(Exception):
    """Base exception for database operations"""

    pass


class BaseDatabase:
    """Base class for database operations with common functionality"""

    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize with database parameters"""
        self.db_params = db_params or DB_PARAMS

    @abstractmethod
    def table_name(self):
        pass

    @abstractmethod
    def pyTable(self):
        pass

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

    def excute_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        return_columns: bool = False,
    ) -> Union[List[tuple], Tuple[List[tuple], List[str]], int]:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    conn.commit()

                    # Check if this is a write operation (INSERT, UPDATE, DELETE)
                    query_upper = query.strip().upper()
                    is_write_op = query_upper.startswith(("INSERT", "UPDATE", "DELETE"))
                    has_returning = "RETURNING" in query_upper

                    # For INSERT/UPDATE operations, try to return the ID
                    if is_write_op:
                        if has_returning and cur.description:
                            # Has RETURNING clause, fetch the returned ID
                            rows = cur.fetchall()
                            if rows:
                                return rows[0][0]  # Return the first column of first row (the ID)
                            return None
                        elif query_upper.startswith(("INSERT", "UPDATE")):
                            # No RETURNING clause, try to get lastval for INSERT or use rowcount
                            if query_upper.startswith("INSERT"):
                                # For INSERT, try to get the last inserted ID
                                cur.execute("SELECT lastval()")
                                result = cur.fetchone()
                                if result:
                                    return result[0]
                            # For UPDATE, return rowcount as indication
                            return cur.rowcount
                        else:
                            # DELETE operation
                            return cur.rowcount

                    # For SELECT queries, fetch rows
                    if cur.description:
                        rows = cur.fetchall()
                        if return_columns:
                            columns = [desc[0] for desc in cur.description]
                            return rows, columns
                        return rows
                    else:
                        # No description means no rows to fetch
                        return []
        except Exception as e:
            raise DatabaseError(f"Failed to execute query: {e}")

    def create_data(self, file, overwrite: bool = False):
        try:
            if isinstance(file, str):
                data = self.pyTable.from_file(file).to_dict()
            elif isinstance(file, self.pyTable):
                data = file.to_dict()
            else:
                raise ValueError(f"Invalid file type: {type(file)}")

            # Check if record already exists (by image_name and process_status_id)
            existing_id = None
            if "image_name" in data and "process_status_id" in data:
                existing = self.read_data_by_params(
                    image_name=data["image_name"], process_status_id=data["process_status_id"]
                )
                if existing:
                    existing_id = (
                        existing if isinstance(existing, int) else (existing[0] if isinstance(existing, list) else None)
                    )

            if overwrite:
                self.delete_data(existing_id)
                existing_id = None

            if existing_id:
                # Update existing record
                self.update_data(existing_id, **data)
                return existing_id
            else:
                # Insert new record
                columns = ", ".join([f'"{k}"' for k in data.keys()])
                values = ", ".join([f"'{v}'" for v in data.values()])

                query = query_insert.format(table_name=self.table_name, columns=columns, values=values)
                id = self.excute_query(query, data)
                return id

        except Exception as e:
            raise DatabaseError(f"Failed to add {self.table_name}: {e}")

    def read_all_data(self, return_type: str = "list", order_by: str = "created_at DESC"):

        if return_type == "table":
            return self.export_to_table(self.table_name, order_by)
        elif return_type == "csv":
            return self.export_to_table(self.table_name, order_by)
        else:
            try:
                # Format query string with table name and order_by using f-string
                query = query_all_columns.format(table_name=self.table_name, order_by=order_by)
                rows, columns = self.excute_query(query, {}, return_columns=True)

                if not rows:
                    return []

                # Convert each row to ProcessStatusTable
                return [self.pyTable.from_row(row, columns=columns) for row in rows]

            except Exception as e:
                raise DatabaseError(f"Failed to read all {self.table_name} data: {e}")

    def read_data(self, name: str):
        """Read image_qa record by name"""
        try:
            # Format query string with table name using f-string
            # Use psycopg parameterization for the value
            query = query_column_by_name.format(table_name=self.table_name)
            params = {"name": name}

            rows, columns = self.excute_query(query, params, return_columns=True)

            if not rows or len(rows) == 0:
                return None

            result = self.pyTable.from_row(rows[0], columns=columns)
            return result

        except Exception as e:
            raise DatabaseError(f"Failed to read {self.table_name} by name: {e}")

    def read_data_by_id(self, target_id: int):
        """Read record by id"""
        try:
            # Format query string with table name using f-string
            # Use psycopg parameterization for the value
            query = query_column_by_id.format(table_name=self.table_name)
            params = {"id": target_id}

            rows, columns = self.excute_query(query, params, return_columns=True)

            if not rows or len(rows) == 0:
                return None

            result = self.pyTable.from_row(rows[0], columns=columns)
            return result

        except Exception as e:
            raise DatabaseError(f"Failed to read {self.table_name} by id: {e}")

    def read_data_by_params(self, return_table=False, **kwargs):

        params = {k: v for k, v in kwargs.items() if v is not None}
        params_str = " AND ".join([f"{k} = %({k})s" for k in params.keys()])
        query = query_by_params.format(table_name=self.table_name, params=params_str)

        rows, columns = self.excute_query(query, params, return_columns=True)

        if not rows or len(rows) == 0:
            return None
        elif len(rows) == 1:
            table = self.pyTable.from_row(rows[0], columns=columns)
            if return_table:
                return table
            else:
                return table.id
        else:
            tables = [self.pyTable.from_row(row, columns=columns) for row in rows]
            if return_table:
                return tables
            else:
                return [table.id for table in tables]

    def read_data_by_params_with_date_range(self, columns, date_min, date_max, **kwargs):
        columns_str = ", ".join(columns)
        params = {k: v for k, v in kwargs.items() if v is not None}
        params["date_min"] = date_min
        params["date_max"] = date_max
        params_str = " AND ".join([f"{k} = %({k})s" for k in params.keys() if k not in ("date_min", "date_max")])
        if params_str:
            params_str += " AND "
        params_str += "date_obs BETWEEN %(date_min)s::timestamp AND %(date_max)s::timestamp"
        query = f"SELECT {columns_str} FROM {self.table_name} WHERE {params_str}"
        rows, column_names = self.excute_query(query, params, return_columns=True)
        return rows

    def update_data(self, target_id: int, **kwargs):
        params = {}
        for k, v in kwargs.items():
            if v is not None:
                # Convert lists/dicts to JSON strings for jsonb columns (warnings, errors)
                if k in ("warnings", "errors") and isinstance(v, (list, dict)):
                    params[k] = json.dumps(v)
                else:
                    params[k] = v
        params_str = ", ".join([f"{k} = %({k})s" for k in params.keys()])
        query = query_update.format(table_name=self.table_name, params=params_str)
        # Use return_rowcount=True for UPDATE statements (they don't return rows)
        id = self.excute_query(query, {**params, "id": target_id})
        return id

    def delete_data(self, target_id: int):
        query = query_delete.format(table_name=self.table_name)
        self.excute_query(query, {"id": target_id})

    def export_to_table(self, order_by: str = "created_at DESC") -> pd.DataFrame:
        """
        Export table data to pandas DataFrame.

        Args:
            table_name: Name of the table to export
            order_by: ORDER BY clause for the query

        Returns:
            pandas DataFrame with the table data
        """
        try:
            # Format query string with table name and order_by using f-string
            query = query_all_columns.format(table_name=self.table_name, order_by=order_by)

            # Use excute_query with return_columns=True to get rows and column names
            rows, columns = self.excute_query(query, return_columns=True)

            if not rows:
                print(f"No data found in {self.table_name}")
                return pd.DataFrame()

            # Create pandas DataFrame
            df = pd.DataFrame(rows, columns=columns)
            return df

        except Exception as e:
            raise DatabaseError(f"Failed to export {self.table_name} to table: {e}")

    def export_to_csv(self, filename: str, order_by: str = "created_at DESC") -> bool:
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
            df = self.export_to_table(order_by)

            if df.empty:
                print(f"No data found in {self.table_name} to export")
                return False

            # Write to CSV file using pandas
            df.to_csv(filename, index=False)

            print(f"Exported {len(df)} records from {self.table_name} to {filename}")
            return True

        except Exception as e:
            raise DatabaseError(f"Failed to export {self.table_name} to CSV: {e}")

    def clear_table(self) -> bool:
        """
        Clear all data from a table.

        Args:
            table_name: Name of the table to clear

        Returns:
            True if successful, False otherwise
        """
        try:
            # Format query string with table name using f-string
            query = query_clear_table.format(table_name=self.table_name)

            # Use excute_query with commit and return_rowcount for write operations
            self.excute_query(query)

            return True

        except Exception as e:
            raise DatabaseError(f"Failed to clear {self.table_name}: {e}")
