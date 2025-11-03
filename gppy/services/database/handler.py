from typing import Optional, Dict, Any, List, Union
from astropy.io import fits
from .pipeline import PipelineDB, PipelineDBError
from .qa import QADB, QADBError
from .table import PipelineData, QAData
import os


class DatabaseHandler:
    """
    High-level database handler that provides a simplified interface
    for database operations used in the preprocessing pipeline.
    Manages both PipelineDB and QADB operations independently.
    """

    def __init__(self, db_params: Optional[Dict[str, Any]] = None, add_database: bool = True):
        """
        Initialize the database handler.

        Args:
            db_params: Database connection parameters
            add_database: Whether to initialize database connection
        """
        self.pipeline_db = PipelineDB(db_params) if add_database else None
        self.qa_db = QADB(db_params) if add_database else None
        self.pipeline_id = None
        self._logger = None

    def set_logger(self, logger):
        """Set logger for database operations"""
        self._logger = logger

    def _log_debug(self, message: str):
        """Log debug message if logger is available"""
        if self._logger:
            self._logger.debug(message)

    def _log_info(self, message: str):
        """Log info message if logger is available"""
        if self._logger:
            self._logger.info(message)

    def _log_warning(self, message: str):
        """Log warning message if logger is available"""
        if self._logger:
            self._logger.warning(message)

    def _log_error(self, message: str):
        """Log error message if logger is available"""
        if self._logger:
            self._logger.error(message)

    # ==================== PIPELINE MANAGEMENT ====================

    def create_pipeline_data(self, config, raw_groups=None, overwrite: bool = False) -> Optional[int]:
        """
        Create a pipeline data in the database and return pipeline ID.

        Args:
            config: Configuration object
            raw_groups: Raw groups data
            overwrite: Whether to overwrite existing records

        Returns:
            Pipeline ID if successful, None otherwise
        """
        if self.pipeline_db is None:
            return None

        try:
            indicators = config.name.split("_")
            if len(indicators) == 3:
                pipe_type = "science"
                obj, filt, run_date = indicators
                unit_name = None
            elif len(indicators) == 2:
                pipe_type = "masterframe"
                run_date, unit_name = indicators
                obj = None
                filt = None
                if raw_groups is None:
                    raise ValueError(f"Raw groups are required for config name: {config.name}")
            else:
                raise ValueError(f"Invalid config name: {config.name}")

            # Find existing pipeline
            existing_pipeline_id = self.pipeline_db._find_existing_pipeline_record(
                run_date=run_date,
                data_type=pipe_type,
                unit=unit_name,
                obj=obj,
                filt=filt,
                config_file=config.config_file if hasattr(config, "config_file") else None,
            )

            if existing_pipeline_id:
                self._log_info(f"Found existing pipeline db record (PID: {existing_pipeline_id})")
                if overwrite:
                    # Use cascade delete to properly handle foreign key constraints
                    self.pipeline_db.delete_pipeline_cascade(existing_pipeline_id)
                    self._log_info(f"Overwriting existing pipeline db record (PID: {existing_pipeline_id})")
                else:
                    self.pipeline_id = existing_pipeline_id
                    config.process_id = existing_pipeline_id
                    self._log_info(f"Using existing pipeline db record (PID: {existing_pipeline_id})")
                    return self.pipeline_id

            if pipe_type == "masterframe":
                # Extract dark and flat info from raw groups
                # group structure: group[0] = [bias_files_list, dark_files_list, flat_files_list]
                dark_info = set([])
                flat_info = set([])
                bias_exists = False

                from ...path.path import PathHandler

                for i, group in enumerate(raw_groups):
                    try:
                        # Check if bias exists in this group
                        # group[0][0] = list of bias files
                        if group and len(group) > 0 and isinstance(group[0], (list, tuple)) and len(group[0]) > 0:
                            if group[0][0]:  # bias files list exists and is non-empty
                                bias_exists = True

                        group_info = PathHandler.get_group_info(group)
                        if ":" in group_info:
                            filt, exptime = group_info.split(":", 1)
                            filt = filt.strip()
                            exptime = exptime.strip()

                            # Check dark files: group[0][1] = list of dark files
                            if group and len(group) > 0 and isinstance(group[0], (list, tuple)) and len(group[0]) > 1:
                                if group[0][1]:  # dark files list exists and is non-empty
                                    # Ensure exptime has 's' suffix
                                    if exptime and not exptime.endswith("s"):
                                        exptime = f"{exptime}s"
                                    dark_info.add(exptime)

                            # Check flat files: group[0][2] = list of flat files
                            if group and len(group) > 0 and isinstance(group[0], (list, tuple)) and len(group[0]) > 2:
                                if group[0][2]:  # flat files list exists and is non-empty
                                    flat_info.add(filt)
                    except Exception as e:
                        self._log_debug(f"Could not parse group {i} info: {e}")

                # Create pipeline data
                pipeline_data = PipelineData.from_config(config, "masterframe")
                # Set bias to True if bias exists, otherwise False
                pipeline_data.bias = bias_exists
                # Set dark as list of exposure times (strings like "10s", "30s", etc.)
                pipeline_data.dark = sorted(list(dark_info)) if dark_info else []
                # Set flat as list of filters (strings like "g", "r", "i", etc.)
                pipeline_data.flat = sorted(list(flat_info)) if flat_info else []

                # Create pipeline record
                self.pipeline_id = self.pipeline_db.create_pipeline_data(pipeline_data)
                self._log_info(f"Created pipeline record with PID: {self.pipeline_id}")
            elif pipe_type == "science":
                pipeline_data = PipelineData.from_config(config, "science")
                self.pipeline_id = self.pipeline_db.create_pipeline_data(pipeline_data)
                self._log_info(f"Created pipeline record with PID: {self.pipeline_id}")
            else:
                raise ValueError(f"Invalid pipe type: {pipe_type}")
            return self.pipeline_id

        except Exception as e:
            self._log_warning(f"Failed to create pipeline record: {e}", exc_info=True)
            self._log_error(f"Failed to create pipeline record: {e}")  # Also log as error for visibility
            self.pipeline_id = None
            return None

    def update_pipeline_progress(self, progress: int, status: str = None) -> bool:
        """
        Update pipeline progress in database.

        Args:
            progress: Progress percentage (0-100)
            status: Optional status string

        Returns:
            True if successful, False otherwise
        """
        if self.pipeline_id is None or self.pipeline_db is None:
            return False

        try:
            if status:
                self.pipeline_db.update_pipeline_data(self.pipeline_id, progress=progress, status=status)
            else:
                self.pipeline_db.update_pipeline_data(self.pipeline_id, progress=progress)
            return True
        except Exception as e:
            self._log_warning(f"Failed to update pipeline progress: {e}")
            return False

    def add_warning(self, count: int = 1) -> bool:
        """Add warning count to pipeline record"""
        if self.pipeline_id is None or self.pipeline_db is None:
            return False

        try:
            return self.pipeline_db.add_warning(self.pipeline_id, count)
        except Exception as e:
            self._log_warning(f"Failed to add warning: {e}")
            return False

    def add_error(self, count: int = 1) -> bool:
        """Add error count to pipeline record"""
        if self.pipeline_id is None or self.pipeline_db is None:
            return False

        try:
            return self.pipeline_db.add_error(self.pipeline_id, count)
        except Exception as e:
            self._log_warning(f"Failed to add error: {e}")
            return False

    # ==================== QA DATA MANAGEMENT ====================
    def create_qa_data(
        self,
        dtype: str,
        image: str = None,
        raw_groups: list = None,
        current_group: int = None,
        key_to_index: dict = None,
        output_file: str = None,
    ) -> None:
        """
        Create QA data in database for a specific data type.

        Args:
            dtype: Data type (bias, dark, flat)
            image: Image file path
            raw_groups: Raw groups data
            current_group: Current group index
            key_to_index: Mapping of data types to indices
            output_file: Output file path for the data type
        """
        if not self.is_connected:
            return

        # Skip QA data creation if pipeline_id is not set
        if self.pipeline_id is None:
            self._log_warning(f"Skipping QA data creation for {dtype}: pipeline_id is not set")
            return

        if dtype == "science":
            header = fits.getheader(image)
            pipe_type = "science"
            output_file = os.path.basename(image)
        elif dtype in ["bias", "dark", "flat"]:
            header = fits.getheader(raw_groups[current_group][1][key_to_index[dtype]])
            pipe_type = "masterframe"
        else:
            return

        qa_data = QAData.from_header(
            header,
            pipe_type,
            f"{dtype}",
            self.pipeline_id,
            os.path.basename(output_file),
        )

        if dtype == "flat":
            trimmed = qa_data.trimmed
        else:
            trimmed = False
        # Create QA data record

        qa_id = self.qa_db.create_qa_data(qa_data)

        if pipe_type == "masterframe" and trimmed:
            self._log_error(
                f"[Group {current_group+1}] {dtype} masterframe is trimmed, a set of masterframes can be trimmed."
            )
            self.add_error()

            with fits.open(raw_groups[current_group][1][key_to_index["bias"]], mode="update") as hdul:
                hdul[0].header["TRIMMED"] = (True, "Non-positive values in the middle of the image")
                hdul[0].header["SANITY"] = (False, "Sanity flag")
                hdul.flush()

            self.update_qa_data("bias", current_group, trimmed=True, sanity=False)

            with fits.open(raw_groups[current_group][1][key_to_index["dark"]], mode="update") as hdul:
                hdul[0].header["TRIMMED"] = (True, "Non-positive values in the middle of the image")
                hdul[0].header["SANITY"] = (False, "Sanity flag")
                hdul.flush()

            self.update_qa_data("dark", current_group, trimmed=True, sanity=False)
        return qa_id

    def update_qa_data(self, dtype: str, current_group: int = None, obj: str = None, **kwargs) -> bool:
        """
        Update QA data record.

        Args:
            dtype: Data type
            current_group: Current group index
            **kwargs: Fields to update

        Returns:
            True if successful, False otherwise
        """
        if self.pipeline_id is None or self.qa_db is None:
            return False

        try:
            # Find the QA record by pipeline_id and dtype
            if dtype == "science":
                qa_type = f"{dtype}_{obj}"
                qa_data = self.qa_db.read_qa_data(pipeline_id=self.pipeline_id, qa_type=qa_type)
            elif dtype in ["bias", "dark", "flat"]:
                qa_type = f"{dtype}_{current_group}"
                qa_data = self.qa_db.read_qa_data(pipeline_id=self.pipeline_id, qa_type=qa_type)
            else:
                raise ValueError(f"Invalid data type: {dtype}")

            if not qa_data:
                self._log_warning(f"No QA data found for {dtype}")
                return False

            # If it's a list, take the first one
            if isinstance(qa_data, list):
                qa_data = qa_data[0]

            # Update using the qa_id
            return self.qa_db.update_qa_data(qa_data.qa_id, **kwargs)
        except Exception as e:
            self._log_warning(f"Failed to update QA data for {dtype}: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.pipeline_db is not None and self.qa_db is not None

    @property
    def has_pipeline_id(self) -> bool:
        """Check if pipeline record exists"""
        return self.pipeline_id is not None

    # ==================== COMBINED OPERATIONS ====================

    def get_pipeline_with_qa(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """Get pipeline data with associated QA data"""
        try:
            # Get pipeline data
            pipeline_record = self.pipeline_db.read_pipeline_data(pipeline_id=pipeline_id)
            if not pipeline_record:
                return None

            # Get associated QA data
            qa_data = self.qa_db.read_qa_data(pipeline_id=pipeline_id)
            if isinstance(qa_data, list):
                qa_list = qa_data
            else:
                qa_list = [qa_data] if qa_data else []

            return {"pipeline": pipeline_record, "qa_data": qa_list, "qa_count": len(qa_list)}

        except Exception as e:
            self._log_error(f"Failed to get pipeline with QA data: {e}")
            return None

    def delete_pipeline_cascade(self, pipeline_id: int) -> bool:
        """Delete pipeline data and all associated QA data"""
        try:
            # Delete QA data first (due to foreign key constraints)
            qa_data = self.qa_db.read_qa_data(pipeline_id=pipeline_id)
            if isinstance(qa_data, list):
                for qa in qa_data:
                    self.qa_db.delete_qa_data(qa.qa_id)

            # Delete pipeline data
            return self.pipeline_db.delete_pipeline_cascade(pipeline_id)

        except Exception as e:
            self._log_error(f"Failed to delete pipeline cascade: {e}")
            return False

    def clear_database(self, passkey=None) -> bool:
        """Clear all data from both pipeline and QA tables"""
        try:
            # Clear QA data first (due to foreign key constraints)
            if passkey == "WANT TO DELETE ALL":
                qa_deleted = self.qa_db.clear_qa_data()

                # Clear pipeline data
                pipeline_deleted = self.pipeline_db.clear_pipeline_data()

                self._log_info(f"Cleared {qa_deleted} QA records and {pipeline_deleted} pipeline records")
            else:
                self._log_info(f"Enter a correct passkey in order to clean up database.")
            return True

        except Exception as e:
            self._log_error(f"Failed to clear database: {e}")
            return False

    def export_to_csv(self, base_filename: str) -> Dict[str, str]:
        """
        Export database tables to CSV files.

        Args:
            base_filename: Base filename without extension (e.g., "2025-01-27_unit1")

        Returns:
            Dict with paths to the exported files:
            {
                'pipeline_data': 'XX_process.csv',
                'qa_data': 'YY_qa.csv'
            }
        """
        try:
            # Generate filenames
            if base_filename.endswith("_process.csv"):
                pipeline_filename = base_filename
                qa_filename = base_filename.replace("_process.csv", "_qa.csv")
            elif base_filename.endswith(".csv"):
                pipeline_filename = base_filename.replace(".csv", "_process.csv")
                qa_filename = base_filename.replace(".csv", "_qa.csv")
            else:
                pipeline_filename = f"{base_filename}_process.csv"
                qa_filename = f"{base_filename}_qa.csv"

            # Export pipeline data
            pipeline_data = self.pipeline_db.export_pipeline_data_to_csv(pipeline_filename)

            # Export QA data
            qa_data = self.qa_db.export_qa_data_to_csv(qa_filename)

            return {"pipeline_data": pipeline_filename, "qa_data": qa_filename}

        except Exception as e:
            self._log_error(f"Failed to export database to CSV: {e}")
            return {}
