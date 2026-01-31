from time import sleep
import gc
import traceback

from pipeline.path import NameHandler
from pipeline.config import PreprocConfiguration
from pipeline.config import SciProcConfiguration
from pipeline.services.database.query import RawImageQuery, free_query
from pipeline.wrapper import DataReduction
from pipeline.services.database import DatabaseHandler
from pipeline.services.blueprint import PreprocessGroup, ScienceGroup


# Run like: python rerun_UDS.py 2>&1 | tee rerun_UDS_tee.log

OVERWRITE_CONFIG = False
# OVERWRITE_DATA = False
OVERWRITE_PREPROCESS = False
OVERWRITE_SCIENCE = False
USE_SYSTEM_QUEUE = True

flist = RawImageQuery().for_target("UDS").image_files()
name = NameHandler(flist)
dates = sorted(set(name.nightdate))

dbh = DatabaseHandler()

for date in dates[::-1]:
    try:
        flist = RawImageQuery().for_target("UDS").on_date(date).image_files()
        print(f"Found {len(flist)} images for {date}")

        dr = DataReduction(list_of_images=flist, use_db=True)
        dr.run(
            overwrite_config=OVERWRITE_CONFIG,
            # overwrite_data=OVERWRITE_DATA,
            overwrite_preprocess=OVERWRITE_PREPROCESS,
            overwrite_science=OVERWRITE_SCIENCE,
            use_system_queue=USE_SYSTEM_QUEUE,
            input_type="Reprocess",
            base_priority=10,
        )

        sleep(120)

    except Exception as e:
        msg = f"Error processing {date}: {e}\n"
        tb = traceback.format_exc()
        print(msg)
        print(tb)
        with open("rerun_UDS_errors.log", "a") as f:
            f.write(msg)
            f.write(tb)

    # break
