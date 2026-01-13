import os
from pathlib import Path
from datetime import datetime
from astropy.io import fits
from ..utils import atleast_1d, collapse
from ..services.database.too import TooDB


# moved out of SciProcConfiguration for readability
def update_too_times(self, input_file):
    """Update ToO database with transfer_time from input file creation time."""

    # Try to find and update the ToO record
    too_db = TooDB()

    input_files = atleast_1d(input_file)

    earliest_time = None
    observation_time = None
    for input_file in input_files:
        if os.path.exists(input_file):
            file_time = datetime.fromtimestamp(os.path.getctime(input_file))
            if earliest_time is None or file_time < earliest_time:
                earliest_time = file_time

            # Parse DATE-OBS - handle ISO format with T separator and milliseconds
            # DATE-OBS in FITS is UTC (as per FITS standard), convert to KST for storage
            try:
                date_obs_str = fits.getval(input_file, "DATE-OBS")
                try:
                    # Try ISO format first (handles 'T' separator and milliseconds)
                    obs_time = datetime.fromisoformat(date_obs_str.replace("Z", "").replace("+00:00", ""))
                except ValueError:
                    # Fall back to space-separated format
                    try:
                        obs_time = datetime.strptime(date_obs_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        # Try with milliseconds
                        obs_time = datetime.strptime(date_obs_str, "%Y-%m-%dT%H:%M:%S.%f")

                # Convert from UTC to KST (Asia/Seoul, UTC+9)
                import pytz

                obs_time_utc = pytz.UTC.localize(obs_time)
                kst = pytz.timezone("Asia/Seoul")
                obs_time = obs_time_utc.astimezone(kst)

                if observation_time is None or obs_time < observation_time:
                    observation_time = obs_time
            except (KeyError, ValueError) as e:
                # DATE-OBS not found or unparseable, skip
                if hasattr(self, "logger") and self.logger:
                    self.logger.debug(f"Could not parse DATE-OBS from {input_file}: {e}")
                continue

    # First, try using config_file if available
    if hasattr(self, "config_file") and self.config_file:
        # Handle case where config_file might be a list
        config_file = self.config_file
        base_path = str(Path(config_file).parent.parent)
        if isinstance(config_file, list):
            config_file = collapse(sorted(config_file)[::-1], force=True) if config_file else None

        if config_file:
            try:
                too_data = too_db.read_data(config_file=config_file)
                if too_data and too_db.too_id:
                    too_db.update_too_data(
                        too_id=too_db.too_id, transfer_time=earliest_time, observation_time=observation_time
                    )
                    too_db.update_too_data(too_id=too_db.too_id, base_path=base_path)
                    if hasattr(self, "logger") and self.logger:
                        self.logger.info(f"Updated ToO transfer_time to {earliest_time.isoformat()}")

                    too_db.send_initial_notice_email(too_db.too_id)

            except Exception as e:
                if hasattr(self, "logger") and self.logger:
                    self.logger.warning(f"Failed to update ToO transfer_time: {e}")
