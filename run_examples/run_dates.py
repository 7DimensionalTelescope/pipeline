import time

from pipeline.wrapper import DataReduction
from pipeline.services.database.query import RawImageQuery, free_query


"""Run like: python run_masterframe_commission_dark.py 2>&1 | tee 2026-02-13_masterframe_commission_dark_tee.log"""


OVERWRITE_SCHEDULE = True
OVERWRITE_CONFIG = True
OVERWRITE_DATA = True
# OVERWRITE_PREPROCESS = False
# OVERWRITE_SCIENCE = False
USE_SYSTEM_QUEUE = True
BASE_PRIORITY = 3

dates = [
    # "2026-04-20",
    "2026-04-21",
    "2026-04-22",
    "2026-04-23",
    "2026-04-24",
    "2026-04-25",
    "2026-04-26",
    "2026-04-27",
    # "2026-04-28",
    "2026-04-29",
]

for date in dates:  # [::-1]:
    print(date)

    flist = RawImageQuery().on_date(date).image_files()  # cross-unit filters exist
    print(f"Found {len(flist)} images for {date}")
    dr = DataReduction(list_of_images=flist, use_db=True, is_pipeline=True)
    dr.run(
        overwrite_config=OVERWRITE_CONFIG,
        overwrite_data=OVERWRITE_DATA,
        # overwrite_preprocess=OVERWRITE_PREPROCESS,
        # overwrite_science=OVERWRITE_SCIENCE,
        overwrite_schedule=OVERWRITE_SCHEDULE,
        use_system_queue=USE_SYSTEM_QUEUE,
        base_priority=BASE_PRIORITY,
    )

    # break

    # prevent fd oversubscription
    time.sleep(120)
