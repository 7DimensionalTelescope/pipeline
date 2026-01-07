import gc
import time
import traceback

from pipeline.wrapper import DataReduction
from pipeline.services.database.query import RawImageQuery, free_query

"""Run like: python run_sciprocess.py 2>&1 | tee 2025-12_sciprocess_tee.log"""

OVERWRITE_CONFIG = False
OVERWRITE_DATA = False
OVERWRITE_SCIENCE = False
USE_SYSTEM_QUEUE = True
SCIPROCESSES = ["astrometry", "photometry", "combine", "subtract"]
error_log_file = "2025-12_sciprocess_errors.log"

query = """
    SELECT DISTINCT date
    FROM survey_night
    WHERE date >= '2025-12-01'
      AND date < '2026-01-01'
    ORDER BY date;
"""

rows = free_query(query, [])
dates = [r[0].strftime("%Y-%m-%d") for r in rows][::-1]
print(dates)

for date in dates:
    dr = None
    try:
        flist = RawImageQuery().on_date(date).image_files()  # cross-unit filters exist
        print(f"Found {len(flist)} images for {date}")
        dr = DataReduction(list_of_images=flist, use_db=True)
        dr.run(
            overwrite_config=OVERWRITE_CONFIG,
            overwrite_data=OVERWRITE_DATA,
            overwrite_science=OVERWRITE_SCIENCE,
            use_system_queue=USE_SYSTEM_QUEUE,
            processes=SCIPROCESSES,
        )

        # to avoid swamping file descriptors
        time.sleep(60)

    except Exception as e:
        msg = f"Error processing {date}: {e}\n"
        tb = traceback.format_exc()
        print(msg)
        print(tb)
        with open(error_log_file, "a") as f:
            f.write(msg)
            f.write(tb)

    finally:
        if dr is not None:
            del dr
        gc.collect()

    # break
