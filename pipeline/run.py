import json
from typing import List

from .config import PreprocConfiguration, SciProcConfiguration
from .const.run import DEFAULT_SCIDATA_PROCESSES
from .const.sciproc import (
    SCIPROCESS_REGISTRY,
    ASTROMETRY_SPEC,
    SINGLE_PHOTOMETRY_SPEC,
    COADD_SPEC,
    COADD_PHOTOMETRY_SPEC,
    SUBTRACTION_SPEC,
    DIFFERENCE_PHOTOMETRY_SPEC,
)
from .errors.errors import EmptyInputAfterSanityRejectionError
from .preprocess import Preprocess
from .astrometry import Astrometry
from .photometry import Photometry
from .imcoadd import ImCoadd
from .subtract import ImSubtract


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
):
    """
    Generate master calibration frames for a specific observation set.

    Master frames are stacked calibration images (like dark, flat, bias) that
    help in reducing systematic errors in scientific observations.

    `master_frame_only` and `calib_types` are parameters rather than hardcoded defaults so a
    masters-only pass can be queued: a regeneration cascade has to rebuild every affected
    master before any science frame is calibrated, and a config that interleaves the two
    cannot express that. They are plain parameters, not `preprocess_kwargs` keys, because the
    scheduler stores kwargs as `str(list)` and parses them back through a quote swap
    (scheduler.py:227, :260), which mangles any embedded JSON -- flags survive, JSON does not.
    """

    try:
        config = PreprocConfiguration(config, is_too=is_too)

        kwargs = {}
        if preprocess_kwargs:
            kwargs = json.loads(preprocess_kwargs)

        # preprocess_kwargs is for the constructor; run() takes only what is named here.
        # Splatting the same dict into both raised TypeError on any constructor-only key.
        prep = Preprocess(
            config,
            use_gpu=use_gpu,
            overwrite=overwrite,
            master_frame_only=kwargs.pop("master_frame_only", master_frame_only),
            calib_types=kwargs.pop("calib_types", calib_types),
            is_too=is_too,
            **kwargs,
        )
        prep.run(device_id=device_id, make_plots=make_plots, dry_run=dry_run)
        del config, prep
    except Exception as e:
        raise e


def _record_auto_sanity(config, sanity: bool = None) -> None:
    """
    Config-level sanity from the run outcome: False when every input was sanity-rejected
    (return code 2), None to clear that once a run gets through. Best-effort, never raises,
    and never overwrites a human verdict (ProcessStatus.set_auto_sanity).
    """
    try:
        if not isinstance(config, SciProcConfiguration):
            return
        if not config.node.settings.is_pipeline or config.node.settings.is_too:
            return

        from .services.database.process_status import ProcessStatus

        ProcessStatus().set_auto_sanity(config.node.name, sanity)
    except Exception as e:
        print(f"[WARNING] Failed to record config sanity: {e}")


def run_scidata_reduction(
    config: SciProcConfiguration | str,
    processes: list[str] = DEFAULT_SCIDATA_PROCESSES,
    overwrite: bool = False,
    is_too: bool = False,
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

        if overwrite:
            # Invalidate the flags for every stage this run will (re)produce, up front,
            # so an interrupted overwrite run never leaves stale downstream flags = True.
            for spec in SCIPROCESS_REGISTRY.specs:
                if spec.name in processes:
                    setattr(config.node.flag, spec.name, False)
            config.write_config()

        if ASTROMETRY_SPEC.name in processes and (not getattr(config.node.flag, ASTROMETRY_SPEC.name) or overwrite):
            ast = Astrometry(config)
            ast.run(overwrite=overwrite)
            del ast
        if SINGLE_PHOTOMETRY_SPEC.name in processes and (not getattr(config.node.flag, SINGLE_PHOTOMETRY_SPEC.name) or overwrite):
            phot = Photometry(config, photometry_mode=SINGLE_PHOTOMETRY_SPEC.photometry_mode, overwrite=overwrite)
            phot.run(overwrite=overwrite)
            del phot
        if COADD_SPEC.name in processes and (not getattr(config.node.flag, COADD_SPEC.name) or overwrite):
            coadd = ImCoadd(config, overwrite=overwrite)
            coadd.run()
            del coadd
        if COADD_PHOTOMETRY_SPEC.name in processes and (not getattr(config.node.flag, COADD_PHOTOMETRY_SPEC.name) or overwrite):
            phot = Photometry(config, photometry_mode=COADD_PHOTOMETRY_SPEC.photometry_mode, overwrite=overwrite)
            phot.run(overwrite=overwrite)
            del phot
        if SUBTRACTION_SPEC.name in processes and (not getattr(config.node.flag, SUBTRACTION_SPEC.name) or overwrite):
            subt = ImSubtract(config, overwrite=overwrite)
            subt.run()
            del subt
        if DIFFERENCE_PHOTOMETRY_SPEC.name in processes and (not getattr(config.node.flag, DIFFERENCE_PHOTOMETRY_SPEC.name) or overwrite):
            phot = Photometry(config, photometry_mode=DIFFERENCE_PHOTOMETRY_SPEC.photometry_mode, overwrite=overwrite)
            phot.run(overwrite=overwrite)
            del phot

        if is_too:
            from .services.database.too import TooDB
            from .too.plotting import make_too_output

            too_db = TooDB()
            too_data = too_db.read_data(config.name)

            if too_data.get("final_notice") == 0:
                make_too_output(too_data.get("id"))
                too_db.send_final_notice_email(too_data.get("id"))

        _record_auto_sanity(config, None)  # inputs got through: drop a stale automatic rejection
        del config

    except EmptyInputAfterSanityRejectionError:
        # Return code 2: not a failure. Record it so automatic reruns skip this config.
        _record_auto_sanity(config, False)
        raise

    except Exception as e:
        raise e


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
