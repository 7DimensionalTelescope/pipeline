import json
import os
from astropy.io import fits
from typing import List, Literal, NamedTuple, Optional
from enum import Enum, auto
from typing import Optional

from ..const.sciproc import (
    ASTROMETRY_SPEC,
    COADD_PHOTOMETRY_SPEC,
    REJECTION_PROCESS_HEADER_KEY,
    SINGLE_PHOTOMETRY_SPEC,
    ProcessSpec,
    SCIPROCESS_REGISTRY,
)
from ..config.utils import get_key
from ..path.name import NameHandler
from .. import const
from .inspection import SANITY_COMMENT, machine_reject
from .database.base import DatabaseError
from .database.handler import database_failure


class SanityAction(Enum):
    TRUST_SANITY_AND_DROP_IMAGE = auto()  # INSPCOMM beside SANITY=F: a human veto
    RECOMPUTE_SANITY = auto()  # admitted: the criteria of the earlier stages pass on the image's values
    DROP_IMAGE = auto()  # excluded in self.input_images


class CurrentStage(Enum):
    REJECTION_STAGE_UNKNOWN = auto()
    BEFORE_REJECTION = auto()
    AT_REJECTION = auto()
    AFTER_REJECTION = auto()
    CURRENT_STAGE_UNKNOWN = auto()


class StartDecision(NamedTuple):
    """A stage start's action on one input and the SANITY (with REJ_PROC) to write; None writes nothing."""

    action: SanityAction
    sanity: Optional[bool] = None
    rej_proc: Optional[str] = None


# TODO: extract out a clean mixin class. There's an instance usage in preprocess.utils.search_with_date_offsets.
class Checker:
    """
    Mixin class that generates, reads, and filters on the "SANITY" key in image headers.

    At a stage's start a human veto or an earlier stage's rejection is trusted; the rest is
    judged on the criteria of the earlier stages.
    """

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

            return self.apply_qa_criteria(header=header, dtype=dtype)

        elif header is not None:
            return self.apply_qa_criteria(header=header, dtype=dtype)

        else:
            raise ValueError("Either file_path or header must be provided")

    def human_rejected(self, file_path: str) -> bool:
        """SANITY F with a human INSPCOMM: a veto."""
        header = fits.getheader(file_path)
        human = "INSPCOMM" in header and header["INSPCOMM"] != "Auto-generated binned master frame"
        return header.get("SANITY") is False and human

    def apply_sanity_filter_and_report(
        self,
        current_process: ProcessSpec,
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
            self.input_images, current_process=current_process, overwrite=overwrite
        )

        filtered_count = len(self.input_images)
        input_has_changed = filtered_count != original_count

        for img, decision in sanity_updates.items():
            try:
                if decision.sanity:
                    with fits.open(img, mode="update") as hdul:
                        hdul[0].header["SANITY"] = (True, SANITY_COMMENT)
                        hdul[0].header.remove(REJECTION_PROCESS_HEADER_KEY, ignore_missing=True)
                        hdul.flush()
                else:
                    machine_reject(img, decision.rej_proc)
                self.logger.debug(f"Updated SANITY header to {decision.sanity} for {os.path.basename(img)}")
                self.sync_sanity_to_image_qa(img, decision.sanity, decision.rej_proc)
            except DatabaseError:
                raise  # ENSURE_DB: sync_sanity_to_image_qa's failure is no header failure
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

    def _filter_by_sanity(
        self,
        images: List[str],
        current_process: ProcessSpec,
        overwrite: bool = False,
    ) -> tuple[List[str], dict[str, StartDecision]]:
        """
        Filter images by `start_decision` on the image_qa row, or on the header when no row decides.

        Returns:
            Tuple of (filtered_images, sanity_updates) where:
            - filtered_images: List of image paths that pass SANITY check
            - sanity_updates: Mapping of image path -> decision whose SANITY should be written back
        """
        if not images:
            return images, {}
        if not isinstance(images, list):
            images = [images]

        # DB fast path: one query instead of one header read per image (~120 ms each on
        # NFS). image_qa holds the recorded verdict and the earlier stages' metrics; skipped
        # on overwrite reruns, which read the headers, and at astrometry.
        names = {image: os.path.basename(image).replace(".fits", "") for image in images}
        db_rows: dict = {}
        row_rejected: set = set()
        trust_db = bool(get_key(getattr(self.config_node, "settings", None), "sanity_from_db", default=True))
        # astrometry parses every input header right after its start check
        trust_db = trust_db and current_process.name != ASTROMETRY_SPEC.name
        # below ~20 images the header reads win: a nightly 3-image stack never depends on
        # the DB at all, so daily reduction keeps zero new failure modes
        if trust_db and not overwrite and len(images) >= 20 and getattr(self, "is_connected", False):
            order = stage_order()
            keys = sorted(
                {
                    key
                    for stage, rules in self.load_qa_criteria(const.IMAGE_GROUP_SCIENCE).items()
                    if order.index(stage) < order.index(current_process.name)
                    for key in rules
                }
            )
            columns = ["image_name", "image_type", "sanity", "inspcomm", "rej_proc"] + [key.lower() for key in keys]
            try:
                from .database.query import free_query

                rows = free_query(
                    f"SELECT {', '.join(columns)} FROM image_qa WHERE image_name = ANY(%s)",
                    (list(names.values()),),
                    statement_timeout_ms=15000,
                )
                # no verdict, or a rejection older than the rej_proc column: the header decides
                db_rows = {
                    row[0]: dict(zip(columns, row))
                    for row in rows
                    if row[2] is not None and not (row[2] is False and row[4] is None)
                }
                row_rejected = {row[0] for row in rows if row[2] is False and row[4] is None}
            except Exception as e:
                if getattr(self, "is_too", False):
                    self.logger.warning(f"image_qa SANITY lookup failed; reading headers instead: {e}")
                else:
                    database_failure(self.logger, f"image_qa SANITY lookup failed; reading headers instead: {e}", e)
            if db_rows:
                self.logger.info(
                    f"SANITY from image_qa for {len(db_rows)}/{len(images)} images -- may lag the "
                    "headers if edited out of band (should not happen in normal operations); "
                    "set settings.sanity_from_db: False to read headers"
                )

        filtered_images = []
        sanity_updates = {}
        for image in images:
            try:
                if (row := db_rows.get(names[image])) is not None:
                    values = {key: row[key.lower()] for key in keys} | {"FILENAME": os.path.basename(image)}
                    decision = self.start_decision(
                        row["sanity"], row["inspcomm"], row["rej_proc"], current_process, values, row["image_type"]
                    )
                else:
                    # Read header once and check sanity
                    header: fits.Header = fits.getheader(image)
                    # has_inspcomm = "INSPCOMM" in header
                    # existing_sanity: bool | None = header.get("SANITY")
                    header["FILENAME"] = os.path.basename(image)
                    if names[image] in row_rejected and "SANITY" not in header:
                        decision = StartDecision(SanityAction.DROP_IMAGE)  # the row's verdict stands
                    else:
                        decision = self.start_decision(
                            header.get("SANITY"),
                            header.get("INSPCOMM"),
                            header.get(REJECTION_PROCESS_HEADER_KEY),
                            current_process,
                            header,
                            NameHandler(image).type.image_type,
                        )

                if decision.sanity is not None:
                    sanity_updates[image] = decision

                match decision.action:
                    case SanityAction.TRUST_SANITY_AND_DROP_IMAGE:
                        self.logger.debug(f"Trusting SANITY=False as-is: {os.path.basename(image)}")
                        continue

                    case SanityAction.DROP_IMAGE:
                        self.logger.debug(f"Filtered out image due to SANITY=False: {os.path.basename(image)}")
                        continue

                    case SanityAction.RECOMPUTE_SANITY:
                        filtered_images.append(image)

                ###

                # if "SANITY" in header and existing_sanity is False:
                #     if has_inspcomm:
                #         self.logger.debug(f"Trusting SANITY=F as INSPCOMM exists: {os.path.basename(image)}")
                #         continue

                #     rej_proc = (header.get(REJECTION_PROCESS_HEADER_KEY) or "").strip()
                #     keep_for_sanity_reevaluation = False

                #     if overwrite and current_process is not None:
                #         if not rej_proc:
                #             # Legacy SANITY=F without REJ_PROC. Assume the rejection
                #             # happened after the current step, so reevaluation is allowed.
                #             keep_for_sanity_reevaluation = True
                #         else:
                #             rej_spec = SCIPROCESS_REGISTRY.get(rej_proc)
                #             if current_process.progress_start <= rej_spec.progress_start:
                #                 keep_for_sanity_reevaluation = True

                #     if keep_for_sanity_reevaluation:
                #         self.logger.debug(
                #             f"Keeping image despite SANITY=False (REJ_PROC={rej_proc}; "
                #             f"overwrite rerun at or before rejection step): {os.path.basename(image)}"
                #         )
                #     else:
                #         self.logger.debug(f"Filtered out image due to SANITY=False: {os.path.basename(image)}")
                #         continue

                # # Trust SANITY as is for images with INSPCOMM (don't recompute from criteria)
                # if has_inspcomm:
                #     self.logger.debug(f"Trusting SANITY=T as INSPCOMM exists: {os.path.basename(image)}")
                #     filtered_images.append(image)
                #     continue

                # # standard sanity compute route
                # header["FILENAME"] = os.path.basename(image)  # to log the name in apply_criteria
                # computed_sanity = self.apply_criteria(header=header, dtype=dtype)

                # if computed_sanity:
                #     filtered_images.append(image)
                #     if overwrite and existing_sanity is False:
                #         sanity_updates[image] = True
                # else:
                #     self.logger.debug(f"Filtered out image due to SANITY=False: {os.path.basename(image)}")
                #     sanity_updates[image] = False

            except Exception as e:
                # If sanity check fails, log warning but include image to avoid breaking pipeline
                self.logger.warning(
                    f"Failed to check SANITY for {os.path.basename(image)}: {e}. " f"Including image in processing."
                )
                filtered_images.append(image)

        return filtered_images, sanity_updates

    def load_qa_criteria(self, category: Literal["masterframe", "science"] = const.IMAGE_GROUP_MASTERFRAME):
        try:
            category = self._dtype_to_category(category)  # in case input isnot category
            criteria_file = os.path.join(const.REF_DIR, "qa", "preprocess.json")
            with open(criteria_file, "r") as f:
                criteria = json.load(f)
            raw = criteria.pop("RAW")
            if category == const.IMAGE_GROUP_SCIENCE:
                criteria_file = os.path.join(const.REF_DIR, "qa", "science.json")
                with open(criteria_file, "r") as f:
                    self.criteria = {const.CONFIG_TYPE_PREPROCESS: raw, **json.load(f)}
            else:
                self.criteria = {dtype: {**raw, **own} for dtype, own in criteria.items()}
            self.loaded_category = category
            return self.criteria
        except FileNotFoundError:
            raise RuntimeError(f"Criteria file not found: {criteria_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in criteria file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load criteria: {e}")

    def apply_qa_criteria(
        self,
        header: fits.Header,
        dtype: Literal["bias", "dark", "flat", "masterframe", "science"] = None,
    ) -> bool:
        """
        Generate a sanity flag based on the criteria.
        Tolerates missing header keys if dtype is "science"

        `dtype` in Preprocess can be directly passed on to this method.

        dtype: BIAS, DARK, FLAT; a stage section, an image type (its judging stages) or SCIENCE (every stage)
        category: masterframe, science - json filenames
        """
        dtype = self._resolve_dtype(dtype=dtype)
        category = self._dtype_to_category(dtype)

        if not hasattr(self, "criteria") or self.criteria is None or self.loaded_category != category:
            self.load_qa_criteria(category=category)

        criteria = _select_criteria(self.criteria, dtype)

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
            msg = f"Rejected {header.get('FILENAME', 'image')} by key {key}={header[key]}: {value['description']}"
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)

        return sanity

    def sync_sanity_to_image_qa(self, image: str, sanity: bool, rej_proc: Optional[str] = None) -> None:
        """Mirror a SANITY flip into image_qa; they are not synced otherwise because self.input_images miss the rejected images."""
        if not getattr(self, "is_connected", False) or getattr(self, "image_qa", None) is None:
            return
        try:
            name = os.path.basename(image).replace(".fits", "")
            qa_id = self.image_qa.read_data_by_params(image_name=name)
            if qa_id is None:
                self.logger.debug(f"No image_qa row for {name}; SANITY={sanity} recorded in the header only")
                return
            self.image_qa.update_data(qa_id, sanity=sanity, rej_proc=rej_proc if sanity is False else "None")
        except Exception as e:
            database_failure(
                self.logger, f"Failed to mirror SANITY={sanity} for {os.path.basename(image)} into image_qa: {e}", e
            )

    def update_sanity_header(
        self,
        header: fits.Header,
        sanity: bool,
        current_process: Optional[ProcessSpec] = None,
    ) -> None:
        """
        Update SANITY-related header cards on an existing header object in place.
        """
        header["SANITY"] = (sanity, SANITY_COMMENT)

        # if sanity:
        #     rej_proc = (header.get(REJECTION_PROCESS_HEADER_KEY) or "").strip()

        #     if not rej_proc:
        #         # Legacy SANITY=F headers may not carry REJ_PROC. Assume the current
        #         # process precedes the missing rejection provenance and clear any stale key.
        #         if REJECTION_PROCESS_HEADER_KEY in header:
        #             del header[REJECTION_PROCESS_HEADER_KEY]
        #     elif current_process is not None:
        #         rej_spec = SCIPROCESS_REGISTRY.get(rej_proc)
        #         if current_process.progress_start <= rej_spec.progress_start:
        #             del header[REJECTION_PROCESS_HEADER_KEY]
        # elif current_process is not None:
        #     header[REJECTION_PROCESS_HEADER_KEY] = (
        #         current_process.name,
        #         "Sci-process that set SANITY to False",
        #     )

        if not sanity:
            if current_process is not None:
                header[REJECTION_PROCESS_HEADER_KEY] = (
                    current_process.name,
                    "Sci-process that set SANITY to False",
                )
            return

        relation = self._current_stage(header, current_process)

        match relation:
            case CurrentStage.REJECTION_STAGE_UNKNOWN:
                if REJECTION_PROCESS_HEADER_KEY in header:
                    del header[REJECTION_PROCESS_HEADER_KEY]

            case CurrentStage.BEFORE_REJECTION | CurrentStage.AT_REJECTION:
                if REJECTION_PROCESS_HEADER_KEY in header:
                    del header[REJECTION_PROCESS_HEADER_KEY]

            case CurrentStage.AFTER_REJECTION | CurrentStage.CURRENT_STAGE_UNKNOWN:
                pass

    def _current_stage(
        self,
        header: fits.Header,
        current_process: Optional[ProcessSpec],
    ) -> CurrentStage:
        rej_proc = (header.get(REJECTION_PROCESS_HEADER_KEY) or "").strip()

        if not rej_proc:
            return CurrentStage.REJECTION_STAGE_UNKNOWN

        if current_process is None:
            return CurrentStage.CURRENT_STAGE_UNKNOWN

        rej_spec = SCIPROCESS_REGISTRY.get(rej_proc)
        if rej_spec is None:
            return CurrentStage.REJECTION_STAGE_UNKNOWN  # or raise, depending on policy

        if current_process.progress_start < rej_spec.progress_start:
            return CurrentStage.BEFORE_REJECTION
        if current_process.progress_start == rej_spec.progress_start:
            return CurrentStage.AT_REJECTION
        return CurrentStage.AFTER_REJECTION

    def start_decision(
        self,
        sanity: Optional[bool],
        inspcomm: Optional[str],
        rej_proc: Optional[str],
        stage: ProcessSpec | str,
        values,
        image_type: str,
    ) -> StartDecision:
        """Veto or earlier REJ_PROC: drop; else the earlier stages' criteria on `values` reject or admit."""
        order = stage_order()
        position = order.index(getattr(stage, "name", stage))
        rej_proc = (rej_proc or "").strip()

        if sanity is False and inspcomm is not None:
            return StartDecision(SanityAction.TRUST_SANITY_AND_DROP_IMAGE)

        # LEAD: a SANITY F with no or an unknown REJ_PROC is judged like a void one
        if sanity is False and rej_proc in order and order.index(rej_proc) < position:
            return StartDecision(SanityAction.DROP_IMAGE)

        for owner in judging_stages(image_type):
            if order.index(owner) < position and not self.apply_qa_criteria(values, dtype=owner):
                return StartDecision(SanityAction.DROP_IMAGE, False, owner)

        # a stage that judges the image at its end leaves a voided rejection to that judgment
        cleared = sanity is False and order[position] not in judging_stages(image_type)
        return StartDecision(SanityAction.RECOMPUTE_SANITY, True if cleared else None)

    def _resolve_dtype(
        self, dtype: Literal["bias", "dark", "flat", "masterframe", "science"] = None
    ) -> Literal["BIAS", "DARK", "FLAT", "MASTERFRAME", "SCIENCE"]:
        """
        Determine the dtype from the provided dtype or self.dtype.
        Returns the specific dtype string (BIAS, DARK, FLAT, SCIENCE).
        """

        resolved = dtype if dtype is not None else getattr(self, "dtype", None)  # dtype when used as a mixin
        return resolved.upper() if resolved is not None else "SCIENCE"

    def _dtype_to_category(
        self, dtype: Literal["BIAS", "DARK", "FLAT", "MASTERFRAME", "SCIENCE"] = None
    ) -> Literal["masterframe", "science"]:
        """Normalize dtype to either 'masterframe' or 'science'."""

        masterframe_types = {"BIAS", "DARK", "FLAT", "MASTERFRAME"}
        if dtype is None:
            return const.IMAGE_GROUP_MASTERFRAME
        return const.IMAGE_GROUP_MASTERFRAME if dtype.upper() in masterframe_types else const.IMAGE_GROUP_SCIENCE

    def _recreate_pathhandler_instance(self):
        """
        Recreate self.path with filtered images to keep it in sync.

        PathHandler will infer working_dir from input files if None.
        """
        self.path = self.path.replace(input=self.input_images)

        # config_node = self.config_node if hasattr(self, "config_node") else self.config.node
        # is_too = get_key(config_node, "settings.is_too", default=False)  # Get is_too from config settings
        # self.path = PathHandler(self.input_images, working_dir=None, is_too=is_too)


_QA_CRITERIA = {}


def stage_order() -> tuple[str, ...]:
    """REJ_PROC stage names in chain order."""
    return (const.CONFIG_TYPE_PREPROCESS, *(spec.name for spec in SCIPROCESS_REGISTRY.specs))


def judging_stages(image_type: str) -> tuple[str, ...]:
    """Stages whose criteria judge an image type, in chain order."""
    if image_type == const.IMAGE_TYPE_SINGLE:
        return (const.CONFIG_TYPE_PREPROCESS, ASTROMETRY_SPEC.name, SINGLE_PHOTOMETRY_SPEC.name)
    if image_type == const.IMAGE_TYPE_COADD:
        return (COADD_PHOTOMETRY_SPEC.name,)
    return ()


def _select_criteria(criteria: dict, dtype: str) -> dict:
    """A master type's section, a stage's, an image type's judging stages', or every stage's for SCIENCE."""
    if dtype.lower() in criteria:
        return criteria[dtype.lower()]
    if dtype.upper() in criteria or dtype.upper() == "MASTERFRAME":
        return criteria[dtype.upper()]
    stages = criteria if dtype.upper() == "SCIENCE" else judging_stages(dtype.lower())
    return {key: rule for stage in stages for key, rule in criteria[stage].items()}


def qa_criterion(key: str, dtype: str = "SCIENCE") -> dict:
    """One `{value, criteria, description}` entry of ref/qa/<category>.json, read once per process: the stages' own floors."""
    category = "masterframe" if dtype.upper() in {"BIAS", "DARK", "FLAT", "MASTERFRAME"} else "science"
    if category not in _QA_CRITERIA:
        _QA_CRITERIA[category] = Checker().load_qa_criteria(category)
    return _select_criteria(_QA_CRITERIA[category], dtype)[key]
