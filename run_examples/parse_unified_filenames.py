"""Parse log file for 'Unified filename not found' entries and extract filenames."""

from pathlib import Path
from typing import List


def parse_unified_filename_errors(log_file_path: str | Path) -> List[str]:
    """
    Parse log file for lines starting with 'Unified filename not found'
    and extract the filenames into a list.

    Parameters
    ----------
    log_file_path : str or Path
        Path to the log file to parse

    Returns
    -------
    List[str]
        List of filenames that were not found (duplicates are included)
    """
    log_file_path = Path(log_file_path)
    filenames = []

    prefix = "Unified filename not found for "

    with open(log_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(prefix):
                # Extract filename (everything after the prefix)
                filename = line[len(prefix) :].strip()
                if filename:  # Only add non-empty filenames
                    filenames.append(filename)

    return filenames


def parse_unified_filename_errors_unique(log_file_path: str | Path) -> List[str]:
    """
    Parse log file for lines starting with 'Unified filename not found'
    and extract unique filenames into a list.

    Parameters
    ----------
    log_file_path : str or Path
        Path to the log file to parse

    Returns
    -------
    List[str]
        List of unique filenames that were not found
    """
    filenames = parse_unified_filename_errors(log_file_path)
    # Return unique filenames while preserving order
    seen = set()
    unique_filenames = []
    for filename in filenames:
        if filename not in seen:
            seen.add(filename)
            unique_filenames.append(filename)
    return unique_filenames


if __name__ == "__main__":
    # Example usage
    log_file = "2026-01-15_masterframe_commission_tee.log"

    print(f"Parsing {log_file}...")
    all_filenames = parse_unified_filename_errors(log_file)
    unique_filenames = parse_unified_filename_errors_unique(log_file)

    print(f"\nTotal entries: {len(all_filenames)}")
    print(f"Unique filenames: {len(unique_filenames)}")
    print(f"\nFirst 10 filenames:")
    for i, filename in enumerate(all_filenames[:10], 1):
        print(f"  {i}. {filename}")

    with open("unified_filenames_not_found.log", "w") as f:
        for filename in unique_filenames:
            f.write(filename + "\n")
