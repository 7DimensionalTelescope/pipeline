import json
import os
from astropy.io import fits
from typing import List, Literal, Optional

from ..const.sciproc import REJECTION_PROCESS_HEADER_KEY, ProcessSpec, SCIPROCESS_REGISTRY
from .. import const


class CheckerMixin:
    """
    Mixin class that generates, reads, and filters on the "SANITY" key in image headers.

    Existing SANITY values are normally trusted, but overwrite-enabled reruns can
    recompute and refresh the header.
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
    ) -> bool:
        """
        Generate a sanity flag based on the criteria.
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

        sanity = True

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
                    sanity = False
                    break
            elif value["criteria"] == "eq":
                if header[key] != value["value"]:
                    sanity = False
                    break
            elif value["criteria"] == "gte":
                if header[key] < value["value"]:
                    sanity = False
                    break
            elif value["criteria"] == "gt":
                if header[key] <= value["value"]:
                    sanity = False
                    break
            elif value["criteria"] == "lte":
                if header[key] > value["value"]:
                    sanity = False
                    break
            elif value["criteria"] == "lt":
                if header[key] >= value["value"]:
                    sanity = False
                    break
            elif value["criteria"] == "within":
                if header[key] < value["value"][0] or header[key] > value["value"][1]:
                    sanity = False
                    break

        if not sanity:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.info(f"Rejected {header.get('FILENAME', 'image')} by key {key}: {value['description']}")
            else:
                print(f"Rejected {header.get('FILENAME', 'image')} by key {key}: {value['description']}")

        return sanity

    def update_sanity_header(
        self,
        header: fits.Header,
        sanity: bool,
        current_process: Optional[ProcessSpec] = None,
    ) -> None:
        """
        Update SANITY-related header cards on an existing header object in place.
        """
        header["SANITY"] = (sanity, "Pipeline image sanity flag")

        if sanity:
            if REJECTION_PROCESS_HEADER_KEY in header:
                del header[REJECTION_PROCESS_HEADER_KEY]
        elif current_process is not None:
            header[REJECTION_PROCESS_HEADER_KEY] = (
                current_process.name,
                "Sci-process that set SANITY to False",
            )

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
            header["FILENAME"] = os.path.basename(file_path)  # to log the name in apply_criteria

            return self.apply_criteria(header=header, dtype=dtype)

        elif header is not None:
            return self.apply_criteria(header=header, dtype=dtype)

        else:
            raise ValueError("Either file_path or header must be provided")

    def _filter_by_sanity(
        self,
        images: List[str],
        dtype: str = "science",
        current_process: Optional[ProcessSpec] = None,
        overwrite: bool = False,
    ) -> tuple[List[str], dict[str, bool]]:
        """
        Filter images based on SANITY check from Checker.

        If SANITY is False in the header, reject the image immediately, unless current_process <= REJ_PROC.
        If SANITY is True or missing, recompute the SANITY from criteria.
        If INSPCOMM is present in the header, trust the SANITY as is and do not
        recompute from criteria, even when overwrite=True.

        Args:
            images: List of image file paths to filter
            dtype: Type of images for sanity check ("science", "masterframe", etc.)

        Returns:
            Tuple of (filtered_images, sanity_updates) where:
            - filtered_images: List of image paths that pass SANITY check
            - sanity_updates: Mapping of image path -> recomputed SANITY value that
              should be written back to the FITS header
        """
        if not images:
            return images, {}
        if not isinstance(images, list):
            images = [images]

        filtered_images = []
        sanity_updates = {}
        for image in images:
            try:
                # Read header once and check sanity
                header: fits.Header = fits.getheader(image)
                has_inspcomm = "INSPCOMM" in header
                existing_sanity: bool | None = header.get("SANITY")

                # Reject if SANITY is False (e.g. for Early QA - header keys may not reproduce early QA)
                if not overwrite and "SANITY" in header and existing_sanity is False:

                    if has_inspcomm:
                        self.logger.debug(f"Trusting SANITY=F as INSPCOMM exists: {os.path.basename(image)}")
                        continue

                    rej_proc = (header.get(REJECTION_PROCESS_HEADER_KEY) or "").strip()

                    keep_for_sanity_reevaluation = False

                    # keep if current stage is before or matches REJ_PROC
                    if current_process is not None and rej_proc:
                        rej_spec = SCIPROCESS_REGISTRY.get(rej_proc)
                        if current_process.progress_start <= rej_spec.progress_start:
                            keep_for_sanity_reevaluation = True

                    if keep_for_sanity_reevaluation:
                        self.logger.debug(
                            f"Keeping image despite SANITY=False (REJ_PROC={rej_proc}; "
                            f"current step precedes or matches rejection step): {os.path.basename(image)}"
                        )
                    else:
                        self.logger.debug(f"Filtered out image due to SANITY=False: {os.path.basename(image)}")
                        continue

                # Trust SANITY as is for images with INSPCOMM (don't recompute from criteria)
                if has_inspcomm:
                    self.logger.debug(f"Trusting SANITY=T as INSPCOMM exists: {os.path.basename(image)}")
                    filtered_images.append(image)
                    continue

                # standard sanity compute route
                header["FILENAME"] = os.path.basename(image)  # to log the name in apply_criteria
                computed_sanity = self.apply_criteria(header=header, dtype=dtype)

                if computed_sanity:
                    filtered_images.append(image)
                    if overwrite and existing_sanity is False:
                        sanity_updates[image] = True
                else:
                    self.logger.debug(f"Filtered out image due to SANITY=False: {os.path.basename(image)}")
                    sanity_updates[image] = False

            except Exception as e:
                # If sanity check fails, log warning but include image to avoid breaking pipeline
                self.logger.warning(
                    f"Failed to check SANITY for {os.path.basename(image)}: {e}. " f"Including image in processing."
                )
                filtered_images.append(image)

        return filtered_images, sanity_updates

    def apply_sanity_filter_and_report(
        self,
        dtype: Literal["science", "masterframe"] = "science",  # Type of images for sanity check
        current_process: Optional[ProcessSpec] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Applies SANITY filtering to `self.input_images` and redefine `self.path`.
        Does not modify config.

        Use this in __init__() after setting self.input_images.
        """
        if not self.input_images:
            return False

        original_count = len(self.input_images)
        original_images = self.input_images.copy()
        self.input_images, sanity_updates = self._filter_by_sanity(
            self.input_images, dtype=dtype, current_process=current_process, overwrite=overwrite
        )

        filtered_count = len(self.input_images)
        input_has_changed = filtered_count != original_count

        for img, sanity_value in sanity_updates.items():
            try:
                with fits.open(img, mode="update") as hdul:
                    self.update_sanity_header(
                        hdul[0].header,
                        sanity_value,
                        current_process=current_process,
                    )
                    hdul.flush()
                    self.logger.debug(f"Updated SANITY header to {sanity_value} for {os.path.basename(img)}")
            except Exception as e:
                self.logger.warning(f"Failed to update SANITY header for {os.path.basename(img)}: {e}")

        if input_has_changed:
            filtered_out = [img for img in original_images if img not in self.input_images]
            self.logger.info(
                f"Filtered {original_count - filtered_count} images based on SANITY check "
                f"({filtered_count}/{original_count} remaining)"
            )
            # Write SANITY=False to header for images rejected by criteria (early-rejected already have it)
            for img in filtered_out:
                self.logger.debug(f"Filtered out image: {os.path.basename(img)}")

            # Recreate self.path to keep it in sync
            self._recreate_pathhandler_instance()

        return input_has_changed

    def _recreate_pathhandler_instance(self):
        """
        Recreate self.path with filtered images to keep it in sync.

        PathHandler will infer working_dir from input files if None.
        """
        self.path = self.path.replace(input=self.input_images)

        # config_node = self.config_node if hasattr(self, "config_node") else self.config.node
        # is_too = get_key(config_node, "settings.is_too", default=False)  # Get is_too from config settings
        # self.path = PathHandler(self.input_images, working_dir=None, is_too=is_too)
