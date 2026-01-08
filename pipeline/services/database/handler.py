from .too import TooDB
from .process_status import ProcessStatus
from .image_qa import ImageQA
import logging


class DatabaseHandler:

    def __init__(
        self,
        db_params=None,
        add_database: bool = True,
        is_too: bool = False,
        logger=None,
    ):
        self.is_too = is_too

        if not hasattr(self, "logger"):
            self.logger = logger or logging.getLogger(__name__)

        if add_database:
            self.too_db = TooDB() if is_too else None
            self.too_id = None
            self.process_status = None if is_too else ProcessStatus(db_params)
            self.process_status_id = None if is_too else None
            self.image_qa = None if is_too else ImageQA(db_params)
            self.image_qa_id = None if is_too else None
        else:
            self.too_db = None
            self.too_id = None
            self.process_status = None
            self.process_status_id = None
            self.image_qa = None
            self.image_qa_id = None

    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        if self.is_too:
            return True
        else:
            return self.process_status is not None and self.image_qa is not None

    def create_process_data(self, config, overwrite: bool = False):

        if self.is_too and self.too_db is not None:
            self.too_db.read_data(config.name)
            self.too_id = self.too_db.too_id
            return None
        elif not self.is_connected:
            return None

        table = self.process_status.pyTable.from_file(config.name)

        existing_process_id = self.process_status.read_data_by_params(**table.to_dict())
        if existing_process_id:
            self.logger.info(f"Found existing process db record (PID: {existing_process_id})")
            if overwrite:
                # Use cascade delete to properly handle foreign key constraints
                try:
                    self.process_status.delete_data(existing_process_id)
                    self.logger.info(f"Deleted existing process db record (PID: {existing_process_id}) for overwrite")
                except Exception as e:
                    self.logger.error(f"Error deleting existing process db record (PID: {existing_process_id}): {e}")
                    raise
            else:
                self.logger.info(f"Using existing process db record (PID: {existing_process_id})")
                self.process_status_id = existing_process_id
                return existing_process_id

        self.process_status_id = self.process_status.create_data(table)
        return self.process_status_id

    def update_progress(self, progress: int, status: str = None) -> bool:
        """
        Update pipeline progress in database.

        Args:
            progress: Progress percentage (0-100)
            status: Optional status string

        Returns:
            True if successful, False otherwise
        """
        if self.is_too and self.too_db is not None:
            return self.too_db.update_too_progress(self.too_id, progress, status)
        elif not self.is_connected:
            return False
        return self.process_status.update_data(self.process_status_id, progress=progress, status=status)

    def create_image_qa_data(self, file: str, process_status_id: int, overwrite: bool = False):

        if not self.is_connected:
            self.logger.warning(f"Skipping QA data creation: process_status_id is not set")
            return None

        table = self.image_qa.pyTable.from_file(file, process_status_id)

        if overwrite:
            self.image_qa.delete_data(table.id)

        qa_id = self.image_qa.create_data(table)

        return qa_id

    def update_image_qa_data(self, image_qa_id: int, data):
        data.pop("id", None)
        self.image_qa.update_data(image_qa_id, **data)
