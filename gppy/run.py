__package__ = "gppy"

import time
from pathlib import Path

from .config import PreprocConfiguration, SciProcConfiguration
from .preprocess import Preprocess
from .astrometry import Astrometry
from .photometry import Photometry
from .imstack import ImStack
from .subtract import ImSubtract

from .services.monitor import Monitor
from .services.queue import QueueManager
from .services.scheduler import Scheduler

from .wrapper import DataReduction


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

        if (not (config.config.flag.astrometry) and "astrometry" in processes) or overwrite:
            astr = Astrometry(config)
            astr.run()
            del astr
        if (not (config.config.flag.single_photometry) and "photometry" in processes) or overwrite:
            phot = Photometry(config)
            phot.run()
            del phot
        if (not (config.config.flag.combine) and "combine" in processes) or overwrite:
            stk = ImStack(config)
            stk.run()
            del stk
        if (not (config.config.flag.combined_photometry) and "photometry" in processes) or overwrite:
            phot = Photometry(config)
            phot.run()
            del phot
        if (not (config.config.flag.subtraction) and "subtract" in processes) or overwrite:
            subt = ImSubtract(config)
            subt.run()
            del subt
        del config
    except Exception as e:
        raise e


def start_monitoring(target_dir=None):
    """
    Initialize and start the astronomical data monitoring system.

    This function:
    1. Creates a QueueManager for parallel task processing
    2. Sets up a Monitor for the raw data directory
    3. Configures a watchdog Observer to track file system changes
    4. Provides a continuous monitoring loop with graceful shutdown

    The monitoring system watches for new calibration and observation data,
    automatically triggering the appropriate processing pipelines.

    Monitoring can be stopped by pressing Ctrl+C.
    """
    if target_dir is None:
        from .const import RAWDATA_DIR

        target_dir = RAWDATA_DIR

    def process_new_images(image_paths):
        queue = QueueManager()
        dr = DataReduction.from_list(image_paths)
        dr.create_config()
        configs = dr.config_list()
        del dr
        sc = Scheduler(*configs)
        queue.add_scheduler(sc)
        queue.wait_until_task_complete("all")

    
    monitor = Monitor(base_path=Path(target_dir))
    monitor.add_callback(process_new_images)
    observer = monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
