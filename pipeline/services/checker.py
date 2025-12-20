import json
import os
from astropy.io import fits
from typing import List
from functools import reduce
from .. import const
from ..path.path import PathHandler
from ..config.utils import get_key


class Checker:
    """
    This class generates and reads the hard-coded "SANITY" key in image headers.
    """

    dtype = None

    def __init__(self, dtype=None):
        self.dtype = dtype
        self.criteria = self.load_criteria()

    def load_criteria(self, dtype="masterframe"):
        try:
            if dtype.upper() in ["BIAS", "DARK", "FLAT", "MASTERFRAME"]:
                dtype = "masterframe"
            else:
                dtype = "science"
            criteria_file = os.path.join(const.REF_DIR, "qa", f"{dtype.lower()}.json")
            with open(criteria_file, "r") as f:
                self.criteria = json.load(f)
                return self.criteria
        except FileNotFoundError:
            raise RuntimeError(f"Criteria file not found: {criteria_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in criteria file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load criteria: {e}")

    def apply_criteria(self, file_path: str = None, header: dict = None, dtype: str = None):
        """
        Generate a sanity flag based on the criteria. Returns the flag and the updated header.
        Tolerates missing header keys if dtype is "science"

        `dtype` in Preprocess can be directly passed on to this method.
        """
        if dtype is None:
            if self.dtype is not None:
                dtype = self.dtype
            else:
                if "bias" in file_path:
                    dtype = "BIAS"
                elif "dark" in file_path:
                    dtype = "DARK"
                elif "flat" in file_path:
                    dtype = "FLAT"
                else:
                    dtype = "SCIENCE"

        if not (hasattr(self, "criteria")):
            self.load_criteria(dtype=dtype)

        criteria = self.criteria[dtype.upper()]

        flag = True

        if header is None:
            if file_path is not None:
                header = fits.getheader(file_path)
            else:
                raise ValueError("Either file_path or header must be provided")

        for key, value in criteria.items():
            if key not in header and dtype.upper() == "SCIENCE":
                continue

            if value["criteria"] == "neq":
                if header[key] == value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "eq":
                if header[key] != value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "gte":
                if header[key] < value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "gt":
                if header[key] <= value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "lte":
                if header[key] > value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "lt":
                if header[key] >= value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "within":
                if header[key] < value["value"][0] or header[key] > value["value"][1]:
                    flag = False
                    break

        header["SANITY"] = (flag, "Sanity flag")

        return flag, header

    def sanity_check(self, file_path: str = None, header: dict = None, dtype: str = None):
        """
        For checking sanity when loading a file from disk.
        """

        if file_path is not None:
            try:
                sanity = fits.getval(file_path, "SANITY")  # assumes boolean value in FITS format
                if sanity is not None:
                    return sanity
            except:
                pass
            flag, _ = self.apply_criteria(file_path=file_path, dtype=dtype)
            return flag

        elif header is not None:
            flag, _ = self.apply_criteria(header=header, dtype=dtype)
            return flag

        else:
            raise ValueError("Either file_path or header must be provided")


class SanityFilterMixin:
    """
    Mixin class that provides SANITY-based input filtering functionality.

    Resets self.path with filtered images, unlike self.config.path, which
    preserves the pristine input images.

    Reused by Photometry, ImStack, and ImSubtract.
    Astrometry always ingest all input images regardless of SANITY check.

    Public API:
        apply_sanity_filter(config_key, update_calibrated=False, dtype="science")
            Main method to filter input images.

    Usage:
        class Photometry(BaseSetup, Checker, SanityFilterMixin):
            def __init__(self, ...):
                super().__init__(config, logger, queue)
                # ... set self.input_images ...
                self.apply_sanity_filter(dtype="science")

    Requires:
    - self.sanity_check() method (from Checker)
    - self.logger attribute
    - self.config.node attribute (ConfigNode)
    - self.path attribute (PathHandler)
    - self.input_images attribute (list of image paths)
    """

    def _filter_by_sanity(self, images: List[str], dtype: str = "science") -> List[str]:
        """
        Filter images based on SANITY check from Checker.

        Args:
            images: List of image file paths to filter
            dtype: Type of images for sanity check ("science", "masterframe", etc.)

        Returns:
            List of image paths that pass SANITY check
        """
        if not images:
            return images
        if not isinstance(images, list):
            images = [images]

        filtered_images = []
        for image in images:
            try:
                # Use Checker's sanity_check method to verify SANITY flag
                sanity = self.sanity_check(file_path=image, dtype=dtype)
                if sanity:
                    filtered_images.append(image)
                else:
                    self.logger.debug(f"Filtered out image due to SANITY=False: {os.path.basename(image)}")
            except Exception as e:
                # If sanity check fails, log warning but include image to avoid breaking pipeline
                self.logger.warning(
                    f"Failed to check SANITY for {os.path.basename(image)}: {e}. " f"Including image in processing."
                )
                filtered_images.append(image)

        return filtered_images

    def apply_sanity_filter_and_report(
        self,
        dtype: str = "science",  # Type of images for sanity check
    ) -> bool:
        """
        Apply SANITY filtering to self.input_images and recreate path.

        This is the main method to call in __init__ after setting input_images.
        Only filters self.input_images and updates self.path - does not modify config.

        Args:
            dtype: Type of images for sanity check ("science", "masterframe", etc.)

        Returns:
            True if input images changed, False otherwise
        """
        if not self.input_images:
            return

        original_count = len(self.input_images)
        original_images = self.input_images.copy()
        self.input_images = self._filter_by_sanity(self.input_images, dtype=dtype)

        filtered_count = len(self.input_images)
        input_changed = filtered_count != original_count
        if input_changed:
            # Track which images were filtered out
            filtered_out = [img for img in original_images if img not in self.input_images]
            self.logger.info(
                f"Filtered {original_count - filtered_count} images based on SANITY check "
                f"({filtered_count}/{original_count} remaining)"
            )
            # Log filtered images at debug level
            for img in filtered_out:
                self.logger.debug(f"Filtered out image: {os.path.basename(img)}")

            # Always recreate path with filtered images to keep it in sync
            self._recreate_path()
        return input_changed

    def _recreate_path(self):
        """
        Recreate self.path with filtered images to keep it in sync.

        PathHandler will infer working_dir from input files if None.
        """
        is_too = get_key(self.config.node, "settings.is_too", default=False)  # Get is_too from config settings
        self.path = PathHandler(self.input_images, working_dir=None, is_too=is_too)
