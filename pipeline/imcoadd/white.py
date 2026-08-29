import os

from astropy.io import fits

from ..const import AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES, WHITE_FILTER
from ..const.crossfilter import CROSSFILTERPROCESS_REGISTRY, WHITE_COADD_SPEC
from ..const.sciproc import COADD_PHOTOMETRY_SPEC, COADD_SPEC
from ..config import CrossFilterConfiguration, SciProcConfiguration
from ..config.utils import get_key
from ..errors import WhiteImageError
from ..path.name import NameHandler
from ..services.database.process_status_dependency import ProcessStatusDependency
from ..utils import add_suffix, atleast_1d, collapse, force_symlink
from .imcoadd import ImCoadd


class WhiteImage(ImCoadd):
    _process_spec = WHITE_COADD_SPEC
    _process_registry = CROSSFILTERPROCESS_REGISTRY
    _process_error = WhiteImageError
    _homogeneous_header_keys = ("OBJECT",)
    _input_label = "coadds"
    _output_filter = WHITE_FILTER
    # PSF summary of the filter coadds: worst-case seeing, mean shape (owner, 2026-08-29)
    _extra_header_keys = ("SEEING", "PEEING", "ELLIP", "ELONG")
    _max_header_keys = ("SEEING", "PEEING")
    _proper_requires_interpolation = False

    @classmethod
    def record_config_dependencies(cls, config_node, logger) -> int:
        """Record declared science-config parents, independent of launch mode.

        White-image parents cannot safely be inferred from image_qa alone:
        historical source coadds do not always have the right process_status
        owner, and sanity-rejected science configs are execution prerequisites
        even though they contribute no pixels.
        """
        if not AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES or not config_node.settings.is_pipeline:
            return 0

        science_configs = list(config_node.input.science_configs or [])
        if not science_configs:
            return 0

        try:
            edges = []
            for parent in science_configs:
                parent_name = NameHandler(parent)
                edges.append((config_node.name, parent_name.stem, parent_name.config_properties["config_type"]))
            return ProcessStatusDependency().replace_dependencies(edges)
        except Exception as exc:
            if logger is None:
                print(f"[WARNING] Failed to record cross-filter config dependencies: {exc}")
            else:
                logger.warning(
                    f"Failed to record cross-filter config dependencies: {exc}",
                    cls._process_error.DependencyRegistrationError,
                )
            return 0

    def sync_config_dependencies(self, process_status_id: int = None) -> int:
        """Use declared parents instead of the generic image-product roll-up."""
        return self.record_config_dependencies(self.config_node, self.logger)

    def initialize(self):
        self._confirm_input_completeness()
        super().initialize()

    def _confirm_input_completeness(self):
        """Coadd only when every available filter is ready: confirmed, proven-rejected, or hold.

        Per declared science parent, ready means flag.coadd and flag.coadd_photometry True and
        the recorded imcoadd.coadd_image on disk without SANITY=False; a parent counts as
        rejected only on config sanity False or FITS SANITY=False evidence; anything else holds
        the run. With is_pipeline, RawFrameQuery is the source of truth for the complete filter
        set. The confirmed list is recorded in node.imcoadd.input_images.
        """
        node = self.config_node
        science_configs = list(node.input.science_configs or [])

        confirmed, rejected, held = [], [], []
        if science_configs:
            for config_file in science_configs:
                try:
                    source = SciProcConfiguration(config_file, write=False, logger=False, verbose=False)
                except Exception as e:
                    raise self._process_error.PrerequisiteNotMetError(f"Unreadable science parent {config_file}: {e}")
                coadd_image = collapse(atleast_1d(get_key(source.node.imcoadd, "coadd_image") or []), force=True)
                complete = bool(get_key(source.node.flag, COADD_SPEC.name, False)) and bool(
                    get_key(source.node.flag, COADD_PHOTOMETRY_SPEC.name, False)
                )
                if get_key(source.node, "sanity") is False or CrossFilterConfiguration._source_rejection_is_proven(
                    source, coadd_image or ""
                ):
                    rejected.append(config_file)
                elif complete and coadd_image and os.path.exists(coadd_image):
                    confirmed.append(coadd_image)
                else:
                    held.append(config_file)
        else:
            for image in list(node.input.expected_coadd_images or []):
                if not os.path.exists(image):
                    held.append(image)
                elif CrossFilterConfiguration._fits_sanity_is_false(image):
                    rejected.append(image)
                else:
                    confirmed.append(image)

        node.input.missing_coadd_images = held
        node.input.sanity_rejected_science_configs = rejected
        if held:
            raise self._process_error.PrerequisiteNotMetError(
                f"{len(held)} input(s) are neither ready nor proven rejected: {held[:3]}"
            )

        used_filters = CrossFilterConfiguration._filters(confirmed) if confirmed else []
        if node.settings.is_pipeline:
            raw_filters = self._raw_filter_inventory()
            covered = set(used_filters) | {
                str(collapse(atleast_1d(NameHandler(entry).filter), force=True)) for entry in rejected
            }
            missing_filters = sorted(raw_filters - covered)
            if missing_filters:
                raise self._process_error.PrerequisiteNotMetError(
                    f"Raw inventory holds filters with no ready or rejected parent here: {missing_filters}"
                )
            extra = sorted(set(used_filters) - raw_filters)
            if extra:
                self.logger.warning(f"Inputs carry filters absent from the raw inventory: {extra}")

        node.input.coadd_images = confirmed
        node.input.used_filters = used_filters
        node.imcoadd.input_images = confirmed

        minimum = int(node.input.minimum_filters or 3)
        if len(used_filters) < minimum:
            raise self._process_error.EmptyInputAfterSanityRejection(
                f"Cross-filter stage found {len(used_filters)} usable filter coadd(s); minimum is {minimum}"
            )
        self.logger.info(f"Confirmed {len(used_filters)} filter coadd(s); {len(rejected)} rejected parent(s)")

    def _raw_filter_inventory(self) -> set:
        from ..services.database import RawFrameQuery

        obj = collapse(atleast_1d(self.path.name.obj), raise_error=True)
        query = RawFrameQuery().for_target(obj)
        if not self.config_node.settings.is_multi_epoch:
            nightdate = collapse(atleast_1d(self.path.name.nightdate), raise_error=True)
            query = query.on_date(nightdate)
        rows = query.fetch()
        if not rows:
            raise self._process_error.PrerequisiteNotMetError(
                f"RawFrameQuery found no frames for {obj}; cannot confirm the complete filter set"
            )
        return {str(row["filter"]) for row in rows if row.get("filter")}

    def apply_sanity_filter_and_report(self, dtype="science", current_process=None, overwrite=False) -> bool:
        """Filter-only: exclude SANITY=False source coadds, never write or recompute their verdicts.

        The white stage is an optional add-on; coadd quality is judged by the
        science chain (coadd_photometry), so it must not stamp SANITY/REJ_PROC
        onto its inputs or un-reject an upstream verdict on overwrite reruns.
        """
        images = list(self.input_images or [])
        if not images:
            return False
        kept = []
        for image in images:
            try:
                sanity = fits.getheader(image).get("SANITY")
            except Exception as e:
                self.logger.warning(f"Could not read SANITY of {os.path.basename(image)}: {e}")
                kept.append(image)
                continue
            if sanity is False:
                self.logger.info(f"Filtered out by SANITY=False: {os.path.basename(image)}")
            else:
                kept.append(image)
        self.input_images = kept
        changed = len(kept) != len(images)
        if changed:
            self.logger.info(f"Sanity filter kept {len(kept)}/{len(images)} source coadds")
            self._recreate_pathhandler_instance()
        return changed

    @classmethod
    def from_list(cls, input_images, working_dir=None):
        for image in input_images:
            if not os.path.exists(image):
                raise FileNotFoundError(f"Input file does not exist: {image}")
        config = CrossFilterConfiguration.user_config(
            input_images=input_images,
            working_dir=working_dir,
            logger=True,
        )
        return cls(config=config)

    def calculate_weight_map(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        use_gpu: bool = True,
        overwrite: bool = False,
        out_weights: list[str] | None = None,
    ) -> list[str]:
        if input_images is None:
            input_images = self.input_images
        input_images = list(atleast_1d(input_images))

        factory = self.path.imcoadd.factory
        out_weights = list(
            atleast_1d(
                out_weights
                if out_weights is not None
                else factory.stage_images(input_images, "weight", factory.bkgsub_dir)
            )
        )
        source_weights = [add_suffix(image, "weight") for image in self.input_images]
        missing = [weight for weight in source_weights if not os.path.exists(weight)]
        if missing:
            raise self._process_error.FileNotFoundError(
                f"Missing {len(missing)} source coadd weight map(s): {missing[:3]}"
            )

        for source, output in zip(source_weights, out_weights):
            if os.path.lexists(output) and not (self.overwrite or overwrite):
                continue
            force_symlink(source, output)

        self.config_node.imcoadd.bkgsub_weight_images = out_weights
        self.logger.info(f"Reused {len(out_weights)} source coadd weight maps for cross-filter weighting")
        return out_weights
