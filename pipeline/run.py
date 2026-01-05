__package__ = "pipeline"

import json
from typing import List

from .config import PreprocConfiguration, SciProcConfiguration
from .preprocess import Preprocess
from .astrometry import Astrometry
from .photometry import Photometry
from .imstack import ImStack
from .subtract import ImSubtract


def run_preprocess(
    config, device_id=None, make_plots=True, overwrite=False, preprocess_kwargs: str = None, is_too=False, use_gpu=False
):
    """
    Generate master calibration frames for a specific observation set.

    Master frames are combined calibration images (like dark, flat, bias) that
    help in reducing systematic errors in scientific observations.
    """

    try:
        # TODO: remove from_config, but for now __init__ causes undefined config_file error
        config = PreprocConfiguration.from_config(config, is_too=is_too)

        kwargs = {}
        if preprocess_kwargs:
            kwargs = json.loads(preprocess_kwargs)

        prep = Preprocess(
            config, use_gpu=use_gpu, overwrite=overwrite, master_frame_only=False, is_too=is_too, **kwargs
        )
        prep.run(device_id=device_id, make_plots=make_plots, **kwargs)
        del config, prep
    except Exception as e:
        raise e


def run_scidata_reduction(
    config: SciProcConfiguration | str,
    processes: list[str] = ["astrometry", "photometry", "combine", "subtract"],
    overwrite: bool = False,
    is_too: bool = False,
):
    # print(config, is_too)
    try:
        if isinstance(config, SciProcConfiguration):
            pass
        elif isinstance(config, str) and config.endswith(".yml"):
            config = SciProcConfiguration(config, is_too=is_too)
        else:
            raise ValueError("Invalid configuration type. Expected SciProcConfiguration or path to .yml file.")

        if config.node.settings.is_too != is_too:
            print(f"[ERROR] is_too mismatch: node.settings.is_too={config.node.settings.is_too} != is_too={is_too}")
            raise ValueError("is_too mismatch")

        if "astrometry" in processes and (not config.node.flag.astrometry or overwrite):
            astr = Astrometry(config)
            astr.run(overwrite=overwrite)
            del astr
        if "photometry" in processes and (not config.node.flag.single_photometry or overwrite):
            phot = Photometry(config, photometry_mode="single_photometry")
            phot.run()
            del phot
        if "combine" in processes and (not config.node.flag.combine or overwrite):
            stk = ImStack(config, overwrite=overwrite)
            stk.run()
            del stk
        if "photometry" in processes and (not config.node.flag.combined_photometry or overwrite):
            phot = Photometry(config, photometry_mode="combined_photometry")
            phot.run()
            del phot
        if "subtract" in processes and (not config.node.flag.subtraction or overwrite):
            subt = ImSubtract(config, overwrite=overwrite)
            subt.run()
            del subt
        if "photometry" in processes and (not config.node.flag.difference_photometry or overwrite):
            phot = Photometry(config, photometry_mode="difference_photometry")
            phot.run()
            del phot

        if is_too:
            from .services.database.too import TooDB

            too_db = TooDB()
            too_data = too_db.read_too_data(config.name)

            if too_data.get("final_notice") == 0:
                too_db.send_final_notice_email(too_data.get("id"))

        del config

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
