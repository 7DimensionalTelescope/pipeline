from gppy.wrapper import DataReduction
from gppy.services.scheduler import Scheduler
from gppy.services.queue import QueueManager
from gppy.services.database.query import free_query

from gppy.services.database import DatabaseHandler

DatabaseHandler().clear_database()
import gc

query = """
    SELECT DISTINCT date
    FROM survey_night
    WHERE date > %s
    ORDER BY date;
"""

rows = free_query(query, ["2024-02-01"])
reprocess_dates = [r[0].strftime("%Y-%m-%d") for r in rows]

queue = QueueManager(max_workers=5)
skip = True
for date in reprocess_dates[::-1]:  # processing backwards for mframe selection
    if date == "2024-11-29":  # start date (inclusive)
        skip = False

    if skip:
        # print(f"Skipping {date}")
        continue
    try:
        dr = DataReduction([date], use_db=True)
        dr.create_config(overwrite=True)
        dr.process_all(processes=[], preprocess_only=True, queue=queue, overwrite=False)
        dr.cleanup()
    except Exception as e:
        msg = f"Error processing {date}: {e}\n"
        print(msg)
        with open("preprocess_errors.log", "a") as f:
            f.write(msg)
        continue
    del dr
    gc.collect()
