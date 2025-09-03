__package__ = "gppy"

import time
from pathlib import Path

from .config import PreprocConfiguration, SciProcConfiguration
from .preprocess import Preprocess
from .astrometry import Astrometry
from .photometry import Photometry
from .imstack import ImStack
from .subtract import ImSubtract


def run_preprocess(config, device_id=None, only_with_sci=False, make_plots=True, **kwargs):
    """
    Generate master calibration frames for a specific observation set.

    Master frames are combined calibration images (like dark, flat, bias) that
    help in reducing systematic errors in scientific observations.

    """
    try:
        config = PreprocConfiguration.from_config(config)
        prep = Preprocess(config, use_gpu=True if device_id is not None else False)
        prep.run(device_id=device_id, make_plots=make_plots, only_with_sci=only_with_sci)
        del config, prep
    except Exception as e:
        raise e


def run_scidata_reduction(config, processes=["astrometry", "photometry", "combine", "subtract"], overwrite=False):
    try:
        if isinstance(config, SciProcConfiguration):
            pass
        elif isinstance(config, str) and config.endswith(".yml"):
            config = SciProcConfiguration.from_config(config)
        else:
            raise ValueError("Invalid configuration type. Expected SciProcConfiguration or path to .yml file.")

        if "astrometry" in processes and (not config.config.flag.astrometry or overwrite):
            astr = Astrometry(config)
            astr.run()
            del astr
        if "photometry" in processes and (not config.config.flag.single_photometry or overwrite):
            phot = Photometry(config)
            phot.run()
            del phot
        if "combine" in processes and (not config.config.flag.combine or overwrite):
            stk = ImStack(config)
            stk.run()
            del stk
        if "photometry" in processes and (not config.config.flag.combined_photometry or overwrite):
            phot = Photometry(config)
            phot.run()
            del phot
        if "subtract" in processes and (not config.config.flag.subtraction or overwrite):
            subt = ImSubtract(config)
            subt.run()
            del subt
        del config
    except Exception as e:
        raise e


def query_observations(input_params, use_db=True, **kwargs):
    from .services.database import RawImageQuery, query_observations_manually

    if use_db:
        try:
            list_of_images = RawImageQuery(input_params).image_files(divide_by_img_type=False)
        except Exception as e:
            print(f"Error querying database: {e}")
            print("Falling back to globbing files from filesystem.")
            list_of_images = query_observations_manually(input_params, **kwargs)
    else:
        list_of_images = query_observations_manually(input_params, **kwargs)
    return list_of_images
