import json
from typing import List

from .config import CrossFilterConfiguration, PreprocConfiguration, SciProcConfiguration
from .const import ENSURE_DB
from .const.run import CONFIG_TYPE_PREPROCESS, DEFAULT_CROSSFILTER_PROCESSES, DEFAULT_SCIDATA_PROCESSES
from .const.crossfilter import CROSSFILTERPROCESS_REGISTRY, PHOT7DS_SPEC, WHITE_COADD_SPEC, WHITE_PHOTOMETRY_SPEC
from .const.sciproc import (
    SCIPROCESS_REGISTRY,
    ASTROMETRY_SPEC,
    SINGLE_PHOTOMETRY_SPEC,
    COADD_SPEC,
    COADD_PHOTOMETRY_SPEC,
    SUBTRACTION_SPEC,
    DIFFERENCE_PHOTOMETRY_SPEC,
)
from .errors import Phot7DSError, WhiteImageError
from .errors.errors import EmptyInputAfterSanityRejectionError
from .services.preflight import check_phot7ds_version
from .services.database.base import DatabaseError
from .services.database.handler import database_failure
from .services.version_check import floor_version, recorded_version, stale_reason
from .preprocess import Preprocess
from .astrometry import Astrometry
from .photometry import Photometry, WhiteCatalog
from .imcoadd import ImCoadd, WhiteImage
from .py7dt import Phot7DS
from .subtract import ImSubtract


def strip_launcher_args(argv: list[str]) -> list[str]:
    """Command-line flags without -config, -make_plots, -input_type, -is_too, -device_id."""
    kwargs, args = [], iter(argv)
    for arg in args:
        if arg in ("-config", "-input_type", "-device_id"):
            next(args, None)
        elif arg not in ("-make_plots", "-is_too"):
            kwargs.append(arg)
    return kwargs


def _ensure_db(config, use_database: bool = True) -> None:
    """ENSURE_DB: a pipeline run that is not ToO starts only with the database on and reachable."""
    settings = config.node.settings
    if not ENSURE_DB or not settings.is_pipeline or settings.is_too:
        return

    from .services.database.query import free_query

    if not use_database:
        raise DatabaseError(f"ENSURE_DB: {config.config_file} was started with the database off")
    try:
        free_query("SELECT 1", statement_timeout_ms=5000)
    except Exception as e:
        raise DatabaseError(f"ENSURE_DB: the database cannot be reached for {config.config_file}: {e}") from e


def run_preprocess(
    config: str,
    device_id=None,
    make_plots=True,
    overwrite=False,
    preprocess_kwargs: str = None,
    is_too=False,
    use_gpu=False,
    master_frame_only=False,
    calib_types=None,
    dry_run=False,
    reprocess_on_better_match=None,
    last_run_kwargs=None,
):
    """
    Generate master calibration frames for a specific observation set.

    Master frames are stacked calibration images (like dark, flat, bias) that
    help in reducing systematic errors in scientific observations.
    """

    try:
        config = PreprocConfiguration(config, is_too=is_too)

        kwargs = {}
        if preprocess_kwargs:
            kwargs = json.loads(preprocess_kwargs)

        # dry_run may arrive via preprocess_kwargs (wrapper.py); a sizing pass must not touch the DB
        dry_run = kwargs.pop("dry_run", dry_run)
        if not dry_run:
            _ensure_db(config, kwargs.get("use_database", True))
            config.node.settings.last_run_kwargs = last_run_kwargs
        prep = Preprocess(
            config,
            use_gpu=use_gpu,
            overwrite=overwrite,
            master_frame_only=kwargs.pop("master_frame_only", master_frame_only),
            calib_types=kwargs.pop("calib_types", calib_types),
            reprocess_on_better_match=kwargs.pop("reprocess_on_better_match", reprocess_on_better_match),
            is_too=is_too,
            use_database=kwargs.pop("use_database", not dry_run),
            **kwargs,
        )
        prep.run(device_id=device_id, make_plots=make_plots, dry_run=dry_run)
        del config, prep
    except Exception as e:
        raise e


def _record_config_sanity(config, sanity: bool = None) -> None:
    """
    Config-level sanity from the run outcome: False when every input was sanity-rejected
    (return code 2), None to clear that once a run gets through. Best-effort, never raises unless ENSURE_DB,
    and never overwrites a human verdict (ProcessStatus.set_config_sanity).
    """
    try:
        if not isinstance(config, SciProcConfiguration | CrossFilterConfiguration):
            return
        config.node.sanity = sanity
        if not config.node.settings.is_pipeline or config.node.settings.is_too:
            return

        from .services.database.process_status import ProcessStatus

        ProcessStatus().set_config_sanity(config.node.name, sanity)
    except DatabaseError as e:
        database_failure(None, f"Failed to record config sanity: {e}", e)
    except Exception as e:
        print(f"[WARNING] Failed to record config sanity: {e}")


def _record_stageless_error(config, error) -> None:
    """An error raised before any stage exists: put its code on the process_status row as a stage's logger would."""
    try:
        config.logger.error(str(error))
        if not config.node.settings.is_pipeline or config.node.settings.is_too:
            return

        from .services.database.handler import DatabaseHandler, ExceptionHandler

        process_status_id = DatabaseHandler(use_database=True, logger=config.logger).create_process_data(config.node)
        if process_status_id is not None:
            ExceptionHandler(process_status_id).add_exception_code("error", error.error_code)
    except Exception as e:
        print(f"[WARNING] Failed to record the error in process_status: {e}")


def _plan_stages(config_node, specs, processes, overwrite, stale, rebuilt, keep_downstream_flags, logger):
    """Registry-order plan: (spec, overwrite) to run, and the specs whose flag is cleared up front."""
    to_run, to_clear, force = [], [], False
    for spec in specs:
        if overwrite:
            trigger = "overwrite"
        elif stale[spec.name]:
            trigger = "stale"
        elif spec.config_section in rebuilt:
            trigger = "rebuilt"
        else:
            trigger = None

        if spec.name not in processes:
            if force and not keep_downstream_flags:
                to_clear.append(spec)  # its product no longer descends from the regenerated input
            elif trigger == "stale":
                logger.warning(f"{spec.name}: {stale[spec.name]}, but not selected; not run")
            continue

        if trigger is None and not force and getattr(config_node.flag, spec.name):
            logger.info(
                f"{spec.name}: flag True, recorded runtime_version "
                f"{recorded_version(config_node, spec.config_section)!r} >= floor "
                f"{floor_version(spec.config_section)!r}; skipped"
            )
            continue

        spec_overwrite = bool(trigger) or force
        to_run.append((spec, spec_overwrite))
        to_clear.append(spec)
        force = force or spec_overwrite
    return to_run, to_clear


def _count_remade_inputs(config) -> int:
    """Daily pipeline config with recorded progress: inputs last registered by Preprocess without SANITY."""
    settings, flag = config.node.settings, config.node.flag
    done = any(getattr(flag, spec.name) for spec in SCIPROCESS_REGISTRY.specs)
    if not settings.is_pipeline or settings.is_too or settings.is_multi_epoch or not done:
        return 0

    from .path import NameHandler
    from .services.database.query import free_query
    from .utils import atleast_1d

    try:
        names = atleast_1d(NameHandler(config.node.input.calibrated_images).stem)
        rows = free_query(
            "SELECT count(*) FROM image_qa q JOIN process_status p ON p.id = q.process_status_id"
            " WHERE q.image_name = ANY(%s) AND p.config_type = %s AND q.sanity IS NULL",
            [names, CONFIG_TYPE_PREPROCESS],
        )
        return rows[0][0]
    except Exception as e:
        database_failure(config.logger, f"Could not check the input singles for a Preprocess re-make; no reset: {e}", e)
        return 0


def _clear_flags(config, specs) -> None:
    # set flag False for the processes this run regenerates and for the trailing ones it invalidates
    for spec in specs:
        setattr(config.node.flag, spec.name, False)


def _run_sciproc_stage(config, spec, overwrite) -> None:
    if spec is ASTROMETRY_SPEC:
        Astrometry(config).run(overwrite=overwrite)
    elif spec is COADD_SPEC:
        ImCoadd(config).run(overwrite=overwrite)
    elif spec is SUBTRACTION_SPEC:
        ImSubtract(config).run(overwrite=overwrite)
    elif spec in (SINGLE_PHOTOMETRY_SPEC, COADD_PHOTOMETRY_SPEC, DIFFERENCE_PHOTOMETRY_SPEC):
        Photometry(config, photometry_mode=spec.photometry_mode).run(overwrite=overwrite)
    else:
        raise ValueError(f"No stage dispatch for {spec.name}")


def run_scidata_reduction(
    config: SciProcConfiguration | str,
    processes: list[str] = DEFAULT_SCIDATA_PROCESSES,
    overwrite: bool = False,
    is_too: bool = False,
    overwrite_config_sections: list[str] = None,
    keep_downstream_flags: bool = False,
    last_run_kwargs: list[str] = None,
):
    try:
        if isinstance(config, SciProcConfiguration):
            pass
        elif isinstance(config, str) and config.endswith(".yml"):
            config = SciProcConfiguration(config, is_too=is_too, overwrite=overwrite)
        else:
            raise ValueError("Invalid configuration type. Expected SciProcConfiguration or path to .yml file.")

        if config.node.settings.is_too != is_too:
            print(f"[ERROR] is_too mismatch: node.settings.is_too={config.node.settings.is_too} != is_too={is_too}")
            raise ValueError("is_too mismatch")
        _ensure_db(config)

        if n_remade := _count_remade_inputs(config):
            config = SciProcConfiguration.reset_config(config.config_file)
            config.logger.warning(
                f"{n_remade} input single(s) were re-made by Preprocess without their previous science cards; config reset"
            )

        specs = SCIPROCESS_REGISTRY.specs
        # before write_config refreshes info.runtime_version
        stale = {spec.name: stale_reason(config.node, spec.config_section) for spec in specs}

        if overwrite_config_sections:
            config.overwrite_config_sections(overwrite_config_sections)
        config.node.settings.last_run_kwargs = last_run_kwargs

        to_run, to_clear = _plan_stages(
            config.node,
            specs,
            processes,
            overwrite,
            stale,
            set(SciProcConfiguration.expand_config_sections(overwrite_config_sections)),
            keep_downstream_flags,
            config.logger,
        )
        _clear_flags(config, to_clear)

        # run processing modules
        for spec, spec_overwrite in to_run:
            _run_sciproc_stage(config, spec, spec_overwrite)

        if is_too:
            from .services.database.too import TooDB

            too_db = TooDB(logger=config.logger)
            too_data = too_db.read_data(config.name)

            if too_data and too_data.get("final_notice") == 0:
                too_db.send_final_notice_email(too_data.get("id"))

        _record_config_sanity(config, None)  # inputs got through: drop a stale automatic rejection
        _autoclean_factory(config)
        del config

    except EmptyInputAfterSanityRejectionError:
        # Return code 2: not a failure. Record it so automatic reruns skip this config.
        _record_config_sanity(config, False)
        raise

    except Exception as e:
        raise e


def _run_crossfilter_stage(config, spec, overwrite) -> None:
    if spec is WHITE_COADD_SPEC:
        WhiteImage(config).run(overwrite=overwrite)
    elif spec is PHOT7DS_SPEC:
        if not getattr(config.node.flag, WHITE_COADD_SPEC.name):
            raise WhiteImageError.PrerequisiteNotMetError("White image must complete before phot7ds photometry")
        Phot7DS(config).run(overwrite=overwrite)
    elif spec is WHITE_PHOTOMETRY_SPEC:
        if not getattr(config.node.flag, WHITE_COADD_SPEC.name):
            raise WhiteImageError.PrerequisiteNotMetError("White image must complete before its source catalog")
        WhiteCatalog(config).run(overwrite=overwrite)
    else:
        raise ValueError(f"No stage dispatch for {spec.name}")


def run_crossfilter_reduction(
    config: CrossFilterConfiguration | str,
    processes: list[str] = DEFAULT_CROSSFILTER_PROCESSES,
    overwrite: bool = False,
    is_too: bool = False,
    keep_downstream_flags: bool = False,
    last_run_kwargs: list[str] = None,
):
    try:
        if isinstance(config, CrossFilterConfiguration):
            pass
        elif isinstance(config, str) and config.endswith(".yml"):
            config = CrossFilterConfiguration(config, overwrite=overwrite)
        else:
            raise ValueError("Expected CrossFilterConfiguration or path to a .yml file")

        if config.node.settings.is_too != is_too:
            raise ValueError(f"is_too mismatch: node.settings.is_too={config.node.settings.is_too} != is_too={is_too}")
        _ensure_db(config)

        specs = CROSSFILTERPROCESS_REGISTRY.specs
        stale = {spec.name: stale_reason(config.node, spec.config_section) for spec in specs}
        config.node.settings.last_run_kwargs = last_run_kwargs

        # Cold-start fallback for configs launched without a scheduler. The
        # same idempotent write runs again after WhiteImage registers its output.
        WhiteImage.record_config_dependencies(config.node, config.logger)

        effective_overwrite = overwrite or bool(config.node.input.parents_changed)

        # WhiteImage.initialize confirms input completeness against the declared parents
        # (and RawFrameQuery when is_pipeline) and records the confirmed inputs.
        to_run, to_clear = _plan_stages(
            config.node, specs, processes, effective_overwrite, stale, set(), keep_downstream_flags, config.logger
        )
        if any(spec is PHOT7DS_SPEC for spec, _ in to_run):
            try:
                check_phot7ds_version()  # fail here, before the white coadd runs, not at the phot7ds stage
            except Phot7DSError as error:
                _record_stageless_error(config, error)
                raise
        _clear_flags(config, to_clear)

        for spec, spec_overwrite in to_run:
            _run_crossfilter_stage(config, spec, spec_overwrite)

        if all(
            getattr(config.node.flag, spec.name) for spec in CROSSFILTERPROCESS_REGISTRY.specs if spec.name in processes
        ):
            config.node.input.parents_changed = False

        _record_config_sanity(config, None)  # inputs got through: drop a stale automatic rejection
        _autoclean_factory(config)
        del config

    except EmptyInputAfterSanityRejectionError:
        # Return code 2: not a failure. Record it so automatic reruns skip this config.
        _record_config_sanity(config, False)
        raise

    except Exception:
        raise


def query_observations(input_params: List[str], use_db=True, master_frame_only=False, **kwargs):
    if use_db:
        try:
            from .services.database import RawImageQuery

            if master_frame_only:
                list_of_images = (
                    RawImageQuery(input_params).of_types(["bias", "dark", "flat"]).image_files(divide_by_img_type=False)
                )
            else:
                list_of_images = RawImageQuery(input_params).image_files(divide_by_img_type=False)
        except Exception as e:
            print(f"Error querying database: {e}")
            print("Falling back to globbing files from filesystem.")
            from .services.database import query_observations_manually

            list_of_images = query_observations_manually(input_params, **kwargs)
    else:
        from .services.database import query_observations_manually

        list_of_images = query_observations_manually(input_params, **kwargs)
    return list_of_images


def _autoclean_factory(config) -> None:
    """Empty the config's factory after a fully successful run: `settings.factory_autoclean`, or the disk policy when unset."""
    import os
    import shutil

    from .config.utils import get_key
    from .const import FACTORY_AUTOCLEAN_DISKS

    try:
        dirs = (
            config.path.crossfilter.factory_dir
            if isinstance(config, CrossFilterConfiguration)
            else config.path.factory_dir
        )
        dirs = [d for d in (dirs if isinstance(dirs, list) else [dirs]) if d]
        enabled = get_key(config.node.settings, "factory_autoclean")
        if enabled is None:
            enabled = any(
                os.path.abspath(d).startswith(disk.rstrip("/") + "/") for d in dirs for disk in FACTORY_AUTOCLEAN_DISKS
            )
        if not enabled:
            return
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            config.logger.info(f"Factory emptied: {d}")
    except Exception as e:
        print(f"[WARNING] Failed to empty the factory: {e}")
