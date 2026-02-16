import gc
import time
import traceback

from pipeline.wrapper import DataReduction
from pipeline.services.database.query import RawImageQuery, free_query


"""Run like: python run_masterframe_commission_dark.py 2>&1 | tee 2026-02-13_masterframe_commission_dark_tee.log"""


OVERWRITE_CONFIG = True
OVERWRITE_DATA = False
OVERWRITE_PREPROCESS = False
OVERWRITE_SCIENCE = False

USE_SYSTEM_QUEUE = True
OVERWRITE_SCHEDULE = True

MASTER_FRAME_ONLY = True
TYPE = "dark"
BASE_PRIORITY = 10

query = """
    SELECT DISTINCT date
    FROM survey_night
    WHERE date < '2024-02-15'
    ORDER BY date;
"""

rows = free_query(query, [])
dates = [r[0].strftime("%Y-%m-%d") for r in rows]  # [::-1]
print(dates)

for date in dates:  # [::-1]:
    dr = None
    try:
        flist = RawImageQuery().on_date(date).of_types([TYPE]).image_files()  # cross-unit filters exist
        print(f"Found {len(flist)} images for {date}")
        dr = DataReduction(list_of_images=flist, use_db=True, master_frame_only=MASTER_FRAME_ONLY)
        dr.run(
            overwrite_config=OVERWRITE_CONFIG,
            overwrite_data=OVERWRITE_DATA,
            overwrite_preprocess=OVERWRITE_PREPROCESS,
            overwrite_science=OVERWRITE_SCIENCE,
            overwrite_schedule=OVERWRITE_SCHEDULE,
            use_system_queue=USE_SYSTEM_QUEUE,
            processes=[],
            base_priority=BASE_PRIORITY,
        )

        # to avoid swamping file descriptors
        time.sleep(120)

    except Exception as e:
        msg = f"Error processing {date}: {e}\n"
        tb = traceback.format_exc()
        print(msg)
        print(tb)
        with open("2026-02-13_masterframe_commission_dark_errors.log", "a") as f:
            f.write(msg)
            f.write(tb)

    finally:
        if dr is not None:
            del dr
        gc.collect()

    break
