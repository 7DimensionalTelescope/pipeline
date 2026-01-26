import json
import os
from astropy.io import fits
from typing import List, Literal
from .. import const


class CheckerMixin:
    """
    Mixin class that generates, reads, and filters on the "SANITY" key in image headers.
    """

    dtype = None

    def _resolve_dtype(
        self, dtype: Literal["bias", "dark", "flat", "masterframe", "science"] = None
    ) -> Literal["BIAS", "DARK", "FLAT", "MASTERFRAME", "SCIENCE"]:
        """
        Determine the dtype from the provided dtype or self.dtype.
        Returns the specific dtype string (BIAS, DARK, FLAT, SCIENCE).
        """

        resolved = dtype if dtype is not None else self.dtype
        return resolved.upper() if resolved is not None else "science"

    def _dtype_to_category(
        self, dtype: Literal["BIAS", "DARK", "FLAT", "MASTERFRAME", "SCIENCE"] = None
    ) -> Literal["masterframe", "science"]:
        """Normalize dtype to either 'masterframe' or 'science'."""

        masterframe_types = {"BIAS", "DARK", "FLAT", "MASTERFRAME"}
        if dtype is None:
            return "masterframe"
        return "masterframe" if dtype.upper() in masterframe_types else "science"

    def load_criteria(self, category: Literal["masterframe", "science"] = "masterframe"):
        try:
            category = self._dtype_to_category(category)  # in case input isnot category
            criteria_file = os.path.join(const.REF_DIR, "qa", f"{category.lower()}.json")
            with open(criteria_file, "r") as f:
                self.criteria = json.load(f)
                self.loaded_category = category
                return self.criteria
        except FileNotFoundError:
            raise RuntimeError(f"Criteria file not found: {criteria_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in criteria file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load criteria: {e}")

    def apply_criteria(
        self,
        header: fits.Header,
        dtype: Literal["bias", "dark", "flat", "masterframe", "science"] = None,
    ) -> tuple[bool, fits.Header]:
        """
        Generate a sanity flag based on the criteria. Returns the flag and the updated header.
        Tolerates missing header keys if dtype is "science"

        `dtype` in Preprocess can be directly passed on to this method.

        dtype: BIAS, DARK, FLAT, SCIENCE - top level json keys
        category: masterframe, science - json filenames
        """
        dtype = self._resolve_dtype(dtype=dtype)
        category = self._dtype_to_category(dtype)

        if not hasattr(self, "criteria") or self.criteria is None or self.loaded_category != category:
            self.load_criteria(category=category)

        criteria = self.criteria[dtype.upper()]

        flag = True

        if header is None:
            raise ValueError("Header must be provided")

        for key, value in criteria.items():
            if key not in header and dtype.upper() == "SCIENCE":
                continue

            # ignores null keys
            if header.get(key) is None:
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

        header["SANITY"] = (flag, "Pipeline image sanity flag")

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
            header = fits.getheader(file_path)
            flag, _ = self.apply_criteria(header=header, dtype=dtype)
            return flag

        elif header is not None:
            flag, _ = self.apply_criteria(header=header, dtype=dtype)
            return flag

        else:
            raise ValueError("Either file_path or header must be provided")

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
        Applies SANITY filtering to `self.input_images` and overrides it.

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
            self._recreate_pathhandler_instance()
        return input_changed

    def _recreate_pathhandler_instance(self):
        """
        Recreate self.path with filtered images to keep it in sync.

        PathHandler will infer working_dir from input files if None.
        """
        self.path = self.path.replace(input=self.input_images)

        # config_node = self.config_node if hasattr(self, "config_node") else self.config.node
        # is_too = get_key(config_node, "settings.is_too", default=False)  # Get is_too from config settings
        # self.path = PathHandler(self.input_images, working_dir=None, is_too=is_too)
