#!/usr/bin/env python3
"""
Standalone script to check the pipeline database table status and save to ECSV.
This script is independent of the Preprocess class and can be run separately.
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add the pipeline module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gppy.services.database import PipelineDatabase
from gppy.services.database.io import PipelineDBError


def get_all_pipeline_records():
    """Get all pipeline records from the database"""
    db = PipelineDatabase()

    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Query to get all pipeline records
                query = """
                    SELECT 
                        id,
                        tag_id,
                        date,
                        data_type,
                        unit,
                        status,
                        progress,
                        bias_exists,
                        dark_filters,
                        flat_filters,
                        warnings,
                        errors,
                        comments,
                        config_file,
                        created_at,
                        updated_at
                    FROM pipeline_process 
                    ORDER BY created_at DESC
                """

                cur.execute(query)
                columns = [desc[0] for desc in cur.description]
                records = cur.fetchall()

                return columns, records

    except Exception as e:
        print(f"Error querying database: {e}")
        return None, None


def save_to_ecsv(columns, records, output_file):
    """Save the pipeline status to ECSV format"""
    try:
        import astropy.table as at

        # Convert records to list of dictionaries
        data = []
        for record in records:
            row = {}
            for i, col in enumerate(columns):
                value = record[i]

                # Handle JSON fields
                if col in ["dark_filters", "flat_filters"] and value:
                    try:
                        import json

                        value = json.loads(value)
                    except:
                        value = str(value)

                # Handle boolean fields
                elif col == "bias_exists":
                    value = bool(value) if value is not None else False

                # Handle numeric fields
                elif col in ["progress", "warnings", "errors", "comments"]:
                    value = int(value) if value is not None else 0

                # Handle UUID fields and other non-serializable types
                elif value is not None:
                    try:
                        # Convert UUID and other complex types to string
                        value = str(value)
                    except:
                        value = str(value) if value is not None else None

                row[col] = value

            data.append(row)

        # Create astropy table
        table = at.Table(data)

        # Add metadata
        table.meta["description"] = "Pipeline Database Status Report"
        table.meta["generated_at"] = datetime.now().isoformat()
        table.meta["total_records"] = len(records)

        # Save to ECSV
        table.write(output_file, format="ascii.ecsv", overwrite=True)
        print(f"Pipeline status saved to: {output_file}")

        return True

    except ImportError:
        print("astropy not available, saving as simple CSV instead...")
        return save_to_csv(columns, records, output_file)
    except Exception as e:
        print(f"Error saving to ECSV: {e}")
        return False


def save_to_csv(columns, records, output_file):
    """Fallback: save as simple CSV"""
    try:
        import csv

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(columns)

            # Write data
            for record in records:
                writer.writerow(record)

        print(f"Pipeline status saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def print_pipeline_summary(columns, records):
    """Print a summary of the pipeline status"""
    if not records:
        print("No pipeline records found.")
        return

    print("\n" + "=" * 100)
    print("PIPELINE DATABASE TABLE STATUS SUMMARY")
    print("=" * 100)
    print(f"Total Records: {len(records)}")

    # Count by status
    status_counts = {}
    unit_counts = {}
    data_type_counts = {}

    for record in records:
        status = record[columns.index("status")] or "unknown"
        unit = record[columns.index("unit")] or "unknown"
        data_type = record[columns.index("data_type")] or "unknown"

        status_counts[status] = status_counts.get(status, 0) + 1
        unit_counts[unit] = unit_counts.get(unit, 0) + 1
        data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1

    print(f"\nStatus Distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print(f"\nUnit Distribution:")
    for unit, count in sorted(unit_counts.items()):
        print(f"  {unit}: {count}")

    print(f"\nData Type Distribution:")
    for data_type, count in sorted(data_type_counts.items()):
        print(f"  {data_type}: {count}")

    # Show recent records
    print(f"\nRecent Records (last 5):")
    for i, record in enumerate(records[:5]):
        record_id = record[columns.index("id")]
        tag_id = record[columns.index("tag_id")]
        date = record[columns.index("date")]
        unit = record[columns.index("unit")]
        status = record[columns.index("status")]
        progress = record[columns.index("progress")]

        print(
            f"  {i+1}. ID: {record_id}, Tag: {tag_id}, Date: {date}, Unit: {unit}, Status: {status}, Progress: {progress}%"
        )

    print("=" * 100)


def main():
    """Main function"""
    print("Checking Pipeline Database Status...")

    # Test database connection
    db = PipelineDatabase()
    if not db.test_connection():
        print("Failed to connect to database. Exiting.")
        return

    # Get all pipeline records
    columns, records = get_all_pipeline_records()

    if columns is None or records is None:
        print("Failed to retrieve pipeline records.")
        return

    # Print summary
    print_pipeline_summary(columns, records)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"pipeline_status_{timestamp}.ecsv"

    if save_to_ecsv(columns, records, output_file):
        print(f"\nDetailed pipeline status saved to: {output_file}")
    else:
        print("\nFailed to save pipeline status to file.")


if __name__ == "__main__":
    main()
