from gppy.wrapper import DataReduction
from gppy.services.scheduler import Scheduler
from gppy.services.queue import QueueManager
from gppy.services.database.query import free_query

import gc

query = """
    SELECT DISTINCT date
    FROM survey_night
    WHERE date >= '2025-08-01'
      AND date < '2025-09-01'
    ORDER BY date;
"""

rows = free_query(query, [])
reprocess_dates = [r[0].strftime("%Y-%m-%d") for r in rows]

queue = QueueManager(max_workers=5)
skip = True
print(reprocess_dates)
for date in reprocess_dates[::-1]:  # processing backwards for mframe selection
    if date == "2025-08-28":  # start date (inclusive)
        skip = False

    if skip:
        continue
    try:
        dr = DataReduction([date], use_db=True)
        dr.create_config(overwrite=True)
        dr.process_all(processes=["astrometry"], queue=queue, overwrite=False)
        dr.cleanup()
    except Exception as e:
        msg = f"Error processing {date}: {e}\n"
        print(msg)
        with open("astrometry_errors.log", "a") as f:
            f.write(msg)
        continue
    del dr
    gc.collect()
