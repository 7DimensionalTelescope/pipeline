from typing import Optional, Dict, Any, List, Union
from astropy.io import fits
from .process import ProcessDB, ProcessDBError
from .qa import QADB, QADBError
from .table import PipelineData, QAData


class DatabaseHandler:
    """
    High-level database handler that provides a simplified interface
    for database operations used in the preprocessing pipeline.
    Manages both ProcessDB and QADB operations independently.
    """

    def __init__(self, db_params: Optional[Dict[str, Any]] = None, add_database: bool = True):
        """
        Initialize the database handler.

        Args:
            db_params: Database connection parameters
            add_database: Whether to initialize database connection
        """
        self.process_db = ProcessDB(db_params) if add_database else None
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

    def create_pipeline_record(self, config, raw_groups, overwrite: bool = False) -> Optional[int]:
        """
        Create a pipeline record in the database and return pipeline ID.

        Args:
            config: Configuration object
            raw_groups: Raw groups data
            overwrite: Whether to overwrite existing records

        Returns:
            Pipeline ID if successful, None otherwise
        """
        if self.process_db is None:
            return None

        try:
            run_date, unit_name = config.name.split("_")

            # Find existing pipeline
            existing_pipeline_id = self.process_db._find_existing_pipeline_record(
                run_date=run_date,
                data_type="masterframe",
                unit=unit_name,
                config_file=config.config_file if hasattr(config, "config_file") else None,
            )

            if existing_pipeline_id:
                self._log_info(f"Found existing pipeline db record (PID: {existing_pipeline_id})")
                if overwrite:
                    # Use cascade delete to properly handle foreign key constraints
                    self.process_db.delete_pipeline_cascade(existing_pipeline_id)
                    self._log_info(f"Overwriting existing pipeline db record (PID: {existing_pipeline_id})")
                else:
                    self.pipeline_id = existing_pipeline_id
                    config.process_id = existing_pipeline_id
                    self._log_info(f"Using existing pipeline db record (PID: {existing_pipeline_id})")
                    return self.pipeline_id

            # Extract dark and flat info from raw groups
            dark_info = set([])
            flat_info = set([])

            from ...path.path import PathHandler

            for i, group in enumerate(raw_groups):
                try:
                    group_info = PathHandler.get_group_info(group)
                    if ":" in group_info:
                        filt, exptime = group_info.split(":", 1)
                        filt = filt.strip()
                        exptime = exptime.strip()
                        if group[0][1]:
                            dark_info.add(exptime)
                        if group[0][2]:
                            flat_info.add(filt)
                except Exception as e:
                    self._log_debug(f"Could not parse group {i} info: {e}")

            # Create pipeline data
            pipeline_data = PipelineData.from_config(config, "masterframe")
            pipeline_data.bias = True if hasattr(config, "bias_input") and config.bias_input else False
            pipeline_data.dark = list(dark_info)
            pipeline_data.flat = list(flat_info)

            # Create pipeline record
            self.pipeline_id = self.process_db.create_pipeline_data(pipeline_data)
            self._log_info(f"Created pipeline record with PID: {self.pipeline_id}")
            return self.pipeline_id

        except Exception as e:
            self._log_warning(f"Failed to create pipeline record: {e}")
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
        if self.pipeline_id is None or self.process_db is None:
            return False

        try:
            if status:
                self.process_db.update_pipeline_data(self.pipeline_id, progress=progress, status=status)
            else:
                self.process_db.update_pipeline_data(self.pipeline_id, progress=progress)
            return True
        except Exception as e:
            self._log_warning(f"Failed to update pipeline progress: {e}")
            return False

    def add_warning(self, count: int = 1) -> bool:
        """Add warning count to pipeline record"""
        if self.pipeline_id is None or self.process_db is None:
            return False

        try:
            return self.process_db.add_warning(self.pipeline_id, count)
        except Exception as e:
            self._log_warning(f"Failed to add warning: {e}")
            return False

    def add_error(self, count: int = 1) -> bool:
        """Add error count to pipeline record"""
        if self.pipeline_id is None or self.process_db is None:
            return False

        try:
            return self.process_db.add_error(self.pipeline_id, count)
        except Exception as e:
            self._log_warning(f"Failed to add error: {e}")
            return False

    # ==================== QA DATA MANAGEMENT ====================

    def create_qa_data(self, dtype: str, header, output_file: str, current_group: int) -> Optional[int]:
        """
        Create QA data record for a specific data type.

        Args:
            dtype: Data type (bias, dark, flat)
            header: FITS header
            output_file: Output file path
            current_group: Current group index

        Returns:
            QA ID if successful, None otherwise
        """
        if self.pipeline_id is None or self.qa_db is None:
            return None

        try:
            self._log_info(f"[Group {current_group+1}] Creating QA data for {dtype} (PID: {self.pipeline_id})")

            qa_data = QAData.from_header(
                header,
                "masterframe",
                f"{dtype}",
                self.pipeline_id,
                output_file,
            )

            # Create QA data record
            qa_id = self.qa_db.create_qa_data(qa_data)
            self._log_debug(f"[Group {current_group+1}] Created QA record with ID: {qa_id}")
            return qa_id

        except Exception as e:
            self._log_error(
                f"[Group {current_group+1}] Failed to create QA data for {dtype} (PID: {self.pipeline_id}): {e}"
            )
            self.add_error()
            return None

    def update_qa_data(self, dtype: str, current_group: int, **kwargs) -> bool:
        """
        Update QA data record.

        Args:
            dtype: Data type (bias, dark, flat)
            current_group: Current group index
            **kwargs: Fields to update

        Returns:
            True if successful, False otherwise
        """
        if self.pipeline_id is None or self.qa_db is None:
            return False

        try:
            # Find the QA record by pipeline_id and dtype
            qa_data = self.qa_db.read_qa_data(pipeline_id_id=self.pipeline_id, qa_type=f"{dtype}_{current_group}")

            if not qa_data:
                self._log_warning(f"No QA data found for {dtype}_{current_group}")
                return False

            # If it's a list, take the first one
            if isinstance(qa_data, list):
                qa_data = qa_data[0]

            # Update using the qa_id
            return self.qa_db.update_qa_data(qa_data.qa_id, **kwargs)
        except Exception as e:
            self._log_warning(f"Failed to update QA data for {dtype}: {e}")
            return False

    def write_qa_data(
        self, dtype: str, raw_groups: list, current_group: int, key_to_index: dict, output_file: str, logger
    ) -> None:
        """
        Write QA data to database for a specific data type.

        Args:
            dtype: Data type (bias, dark, flat)
            raw_groups: Raw groups data
            current_group: Current group index
            key_to_index: Mapping of data types to indices
            output_file: Output file path for the data type
            logger: Logger instance
        """
        if not self.is_connected:
            return

        try:
            header = fits.getheader(raw_groups[current_group][1][key_to_index[dtype]])
        except:
            return

        try:
            qa_data = QAData.from_header(
                header,
                "masterframe",
                f"{dtype}",
                self.pipeline_id,
                output_file,
            )

            trimmed = qa_data.trimmed
            # Create QA data record
            qa_id = self.create_qa_data(dtype, header, output_file, current_group)
        except Exception as e:
            self._log_error(
                f"[Group {current_group+1}] Failed to create QA data for {dtype} (PID: {self.pipeline_id}): {e}"
            )
            self.add_error()

        if trimmed:
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

    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.process_db is not None and self.qa_db is not None

    @property
    def has_pipeline_id(self) -> bool:
        """Check if pipeline record exists"""
        return self.pipeline_id is not None

    # ==================== COMBINED OPERATIONS ====================

    def get_pipeline_with_qa(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """Get pipeline data with associated QA data"""
        try:
            # Get pipeline data
            pipeline_record = self.process_db.read_pipeline_data(pipeline_id=pipeline_id)
            if not pipeline_record:
                return None

            # Get associated QA data
            qa_data = self.qa_db.read_qa_data(pipeline_id_id=pipeline_id)
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
            qa_data = self.qa_db.read_qa_data(pipeline_id_id=pipeline_id)
            if isinstance(qa_data, list):
                for qa in qa_data:
                    self.qa_db.delete_qa_data(qa.qa_id)

            # Delete pipeline data
            return self.process_db.delete_pipeline_cascade(pipeline_id)

        except Exception as e:
            self._log_error(f"Failed to delete pipeline cascade: {e}")
            return False

    def clear_database(self) -> bool:
        """Clear all data from both pipeline and QA tables"""
        try:
            # Clear QA data first (due to foreign key constraints)
            qa_deleted = self.qa_db.clear_qa_data()

            # Clear pipeline data
            pipeline_deleted = self.process_db.clear_pipeline_data()

            self._log_info(f"Cleared {qa_deleted} QA records and {pipeline_deleted} pipeline records")
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
            pipeline_data = self.process_db.export_pipeline_data_to_csv(pipeline_filename)

            # Export QA data
            qa_data = self.qa_db.export_qa_data_to_csv(qa_filename)

            return {"pipeline_data": pipeline_filename, "qa_data": qa_filename}

        except Exception as e:
            self._log_error(f"Failed to export database to CSV: {e}")
            return {}
