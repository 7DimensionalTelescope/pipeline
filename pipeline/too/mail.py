import os
import numpy as np
import pytz

from ..services.database.too import TooDB, TooDBError


class TooMail:
    """Email notification handler for ToO requests."""

    def __init__(self, too_db: TooDB):
        """
        Initialize the mail handler with a TooDB instance.

        Args:
            too_db: TooDB instance for database operations
        """
        self.too_db = too_db

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
        from ..const import EMAIL_PASSWORD, EMAIL_USER

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
        from ..const import DEFAULT_RECIPIENT

        too_data = self.too_db.read_data_by_id(too_id)

        if too_data.get("init_notice") == 1:
            return True
        else:
            self.too_db.update_too_data(too_id=too_id, init_notice=1)

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
        from ..const import DEFAULT_RECIPIENT

        self.too_db.update_too_data(too_id=too_id, final_notice=1)

        too_data = self.too_db.read_data_by_id(too_id)

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
        from .table import format_sed_table_string

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
        from ..const import DEFAULT_RECIPIENT

        too_data = self.too_db.read_data_by_id(too_id)

        if too_data.get("num_filters") == 0:
            num_filters = self._count_num_filters(too_id)
        else:
            num_filters = too_data.get("num_filters")

        if too_data.get("interim_notice") == 1:
            return True
        else:
            self.too_db.update_too_data(too_id=too_id, interim_notice=1)

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

        self.too_db.update_too_data(too_id=too_id, interim_notice=1)

        return True

    def _build_interim_notice_contents(
        self,
        too_data,
        sed_data,
        dtype="difference",
        is_detected=None,
        mag=None,
        mag_error=None,
        is_upper_limit=False,
        num_filters=None,
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
        too_data = self.too_db.read_data_by_id(too_id)
        base_path = too_data.get("base_path")

        if not base_path or not os.path.exists(base_path):
            self.too_db.update_too_data(too_id=too_id, num_filters=0)
            return 0

        # Count only directories that look like filter names
        # Filter names are: u, g, r, i, z (broadband) or mXXX (medium band)
        from ..const import ALL_FILTERS

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

        self.too_db.update_too_data(too_id=too_id, num_filters=num_filters)
        return num_filters
