from gppy.wrapper import DataReduction
from gppy.services.scheduler import Scheduler
from gppy.services.queue import QueueManager
from gppy.services.database.query import free_query

import gc

queue = QueueManager(max_workers=5)

# date = "2025-08-04"
date = "2025-08-19"

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
del dr
gc.collect()
