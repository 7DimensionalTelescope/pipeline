from __future__ import annotations

import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

from astropy.io import fits

from .. import __version__
from ..const.crossfilter import CROSSFILTERPROCESS_REGISTRY
from ..path.name import NameHandler
from ..path.path import CrossFilterPathHandler, PathHandler
from ..services.logger import Logger
from ..utils import atleast_1d, collapse, time_diff_in_seconds
from .base import BaseConfig
from .sciprocess import SciProcConfiguration
from .utils import get_key

if TYPE_CHECKING:
    from ._crossfilter_stubs import CrossFilterNode


class CrossFilterConfiguration(BaseConfig):
    if TYPE_CHECKING:
        node: CrossFilterNode

    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger: bool | Logger = None,
        write: bool = True,
        verbose: bool = True,
        overwrite: bool = False,
        working_dir: str | None = None,
        config_file: str | None = None,
        is_pipeline: bool = False,
        is_too: bool = False,
        is_multi_epoch: bool = False,
        config_suffix: str | None = None,
        factory_scratch: str | None = None,
        **kwargs,
    ):
        st = time.time()
        self.write = write
        logger = None if logger is True else logger
        self._handle_input(
            input,
            logger=logger,
            verbose=verbose,
            working_dir=working_dir,
            config_file=config_file,
            is_pipeline=is_pipeline,
            is_too=is_too,
            is_multi_epoch=is_multi_epoch,
            config_suffix=config_suffix,
            factory_scratch=factory_scratch,
            overwrite=overwrite,
            **kwargs,
        )

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize(
                is_pipeline=is_pipeline,
                is_too=is_too,
                is_multi_epoch=is_multi_epoch,
                config_suffix=config_suffix,
                factory_scratch=factory_scratch,
            )
            self.logger.info(f"'CrossFilterConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file: {os.path.basename(self.config_file)}")
            self.logger.debug(f"Full path to the configuration file: {self.config_file}")

        self.fill_missing_from_yaml(self.path.crossfilter_base_yml)
        if not os.path.exists(self.config_file) or overwrite:
            self.write_config()
        self.logger.info("Completed to load configuration")

    @property
    def name(self):
        if hasattr(self, "config_file") and self.config_file is not None:
            return os.path.splitext(os.path.basename(self.config_file))[0]
        if hasattr(self, "path"):
            return os.path.splitext(os.path.basename(self.path.crossfilter.output_yml))[0]
        return get_key(getattr(self, "node", None), "name")

    @staticmethod
    def _science_config_coadd(config_file: str) -> str:
        config = SciProcConfiguration(config_file, write=False, logger=False)
        configured = get_key(config.node.imcoadd, "coadd_image")
        if not configured:
            raise ValueError(f"Science config has no imcoadd.coadd_image: {config_file}")
        return collapse(atleast_1d(configured), raise_error=True)

    @classmethod
    def _science_config_coadds(cls, config_files: list[str]) -> list[str]:
        return [cls._science_config_coadd(path) for path in config_files]

    def _handle_input(
        self,
        input,
        logger,
        verbose,
        working_dir=None,
        config_file=None,
        is_pipeline=False,
        is_too=False,
        is_multi_epoch=False,
        config_suffix=None,
        factory_scratch=None,
        overwrite=False,
        **kwargs,
    ):
        if isinstance(input, list):
            values = sorted(os.path.abspath(os.fspath(value)) for value in input)
            if not values:
                raise ValueError("Cross-filter configuration requires at least one input")
            if all(value.endswith((".yml", ".yaml")) for value in values):
                self.science_configs = values
                self.input_files = self._science_config_coadds(values)
            elif all(value.endswith(".fits") for value in values):
                self.science_configs = []
                self.input_files = values
            else:
                raise ValueError("Cross-filter inputs must be all science configs or all FITS images")

            self.path = CrossFilterPathHandler(
                self.input_files,
                working_dir=working_dir,
                is_pipeline=is_pipeline,
                is_too=is_too,
                is_multi_epoch=is_multi_epoch,
                config_file=config_file,
                config_suffix=config_suffix,
                factory_scratch=factory_scratch,
            )
            config_source = self.path.crossfilter_base_yml
            self.config_file = self.path.crossfilter.output_yml
            log_file = self.path.crossfilter.output_log
            if self.write and os.path.exists(self.config_file) and not overwrite:
                raise FileExistsError(
                    f"Cross-filter config already exists: {self.config_file}. "
                    "Load that YAML, pass overwrite=True, or choose config_suffix for a separate config."
                )
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=log_file,
                verbose=verbose,
                overwrite=self.write,
            )
            self.logger.info("Generating 'CrossFilterConfiguration' from the base configuration")
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.node.logging.file = log_file
        elif isinstance(input, str | dict):
            config_source = input
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.path = self._set_pathhandler_from_config(
                working_dir=working_dir,
                is_pipeline=get_key(self.node.settings, "is_pipeline", False),
                is_too=get_key(self.node.settings, "is_too", False),
                is_multi_epoch=get_key(self.node.settings, "is_multi_epoch", False),
                config_suffix=get_key(self.node.settings, "config_suffix"),
                factory_scratch=get_key(self.node.settings, "factory_scratch"),
                config_file=config_source if isinstance(config_source, str) else None,
            )
            if isinstance(config_source, str):
                self.config_file = config_source
            else:
                self.config_file = self.path.crossfilter.output_yml
            self.node.logging.file = self.path.crossfilter.output_log
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.node.logging.file if self.write else None,
                verbose=verbose,
                overwrite=overwrite,
            )
            self._initialized = True
            self.logger.info("Loading 'CrossFilterConfiguration' from an existing file or dictionary")
        else:
            raise ValueError("Input must be a list of science configs/FITS images, a config path, or a dictionary")

    def _set_pathhandler_from_config(
        self,
        working_dir=None,
        is_pipeline=False,
        is_too=False,
        is_multi_epoch=False,
        config_suffix=None,
        factory_scratch=None,
        config_file=None,
    ):
        science_configs = list(get_key(self.node.input, "science_configs", []) or [])
        expected = list(get_key(self.node.input, "expected_coadd_images", []) or [])
        if science_configs:
            expected = self._science_config_coadds(science_configs)
        if not expected:
            expected = list(get_key(self.node.input, "coadd_images", []) or [])
        if not expected:
            raise ValueError("Cross-filter configuration has no source coadd images")
        self.science_configs = science_configs
        self.input_files = expected
        return CrossFilterPathHandler(
            expected,
            working_dir=working_dir,
            is_pipeline=is_pipeline,
            is_too=is_too,
            is_multi_epoch=is_multi_epoch,
            config_file=config_file,
            config_suffix=config_suffix,
            factory_scratch=factory_scratch,
        )

    def initialize(
        self,
        is_pipeline=False,
        is_too=False,
        is_multi_epoch=False,
        config_suffix=None,
        factory_scratch=None,
    ):
        self.node.info.creation_version = __version__
        self.node.info.creation_datetime = datetime.now().isoformat()
        self.node.info.file = self.config_file
        self.node.name = self.node.name or self.name
        self.node.settings.is_pipeline = is_pipeline
        self.node.settings.is_too = is_too
        self.node.settings.is_multi_epoch = is_multi_epoch
        self.node.settings.config_suffix = config_suffix
        self.node.settings.factory_scratch = factory_scratch
        self.node.input.science_configs = self.science_configs
        self.node.input.expected_coadd_images = self.input_files
        self.node.input.filters = self._filters(self.input_files)
        self.node.input.source_raw_images = []
        self.node.input.discovery_method = (
            "explicit_science_configs" if self.science_configs else "explicit_coadd_images"
        )
        self.node.input.discovery_datetime = datetime.now().isoformat()
        self.node.input.output_dir = self.path.output_dir
        self.node.input.white_image = self.path.crossfilter.white_image
        self.node.input.white_catalog = self.path.crossfilter.source_catalog
        self.node.imcoadd.coadd_image = self.path.crossfilter.white_image
        self._initialized = True

    @staticmethod
    def _fits_sanity_is_false(image: str) -> bool:
        try:
            value = fits.getval(image, "SANITY")
        except Exception:
            return False
        return value is not None and not bool(value)

    @classmethod
    def _source_rejection_is_proven(cls, source_config: SciProcConfiguration, coadd_image: str) -> bool:
        if os.path.exists(coadd_image):
            return cls._fits_sanity_is_false(coadd_image)

        singles = list(get_key(source_config.node.input, "calibrated_images", []) or [])
        return (
            bool(singles)
            and all(os.path.exists(image) for image in singles)
            and all(cls._fits_sanity_is_false(image) for image in singles)
        )

    def set_science_configs(self, config_files: list[str]) -> None:
        science_configs = sorted(os.path.abspath(os.fspath(path)) for path in config_files)
        expected = self._science_config_coadds(science_configs)
        changed = list(get_key(self.node.input, "science_configs", []) or []) != science_configs
        self.science_configs = science_configs
        self.input_files = expected
        self.node.input.science_configs = science_configs
        self.node.input.expected_coadd_images = expected
        self.node.input.filters = self._filters(expected)
        if changed:
            for spec in CROSSFILTERPROCESS_REGISTRY.specs:
                setattr(self.node.flag, spec.name, False)
            self.node.input.coadd_images = []
            self.node.input.used_filters = []
            self.node.input.sanity_rejected_science_configs = []
            self.node.input.missing_coadd_images = []
            self.node.input.parents_changed = True
            self.node.imcoadd.input_images = []
            self.node.photometry.input_images = []

    def record_discovery(self, raw_images: list[str] | None, method: str) -> None:
        raw_images = sorted(os.path.abspath(os.fspath(path)) for path in (raw_images or []))
        if raw_images:
            raw_filters = sorted(set(str(value) for value in atleast_1d(NameHandler(raw_images).filter)))
            if raw_filters != list(self.node.input.filters or []):
                raise ValueError(
                    f"Raw inventory filters {raw_filters} do not match source-config filters "
                    f"{list(self.node.input.filters or [])}"
                )
        current = list(get_key(self.node.input, "source_raw_images", []) or [])
        if current == raw_images and get_key(self.node.input, "discovery_method") == method:
            return
        self.node.input.source_raw_images = raw_images
        self.node.input.discovery_method = method
        self.node.input.discovery_datetime = datetime.now().isoformat()

    @classmethod
    def user_config(cls, input_images: list[str] | str = None, **kwargs):
        return cls(list(atleast_1d(input_images)), **kwargs)

    @staticmethod
    def _filters(images: list[str]) -> list[str]:
        filters = [str(value) for value in atleast_1d(NameHandler(images).filter)]
        duplicates = sorted({value for value in filters if filters.count(value) > 1})
        if duplicates:
            raise ValueError(f"Cross-filter inputs must contain one coadd per filter; duplicates: {duplicates}")
        return sorted(filters)
