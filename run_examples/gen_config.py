from pipeline.wrapper import DataReduction
from pipeline.services.database.query import RawImageQuery, free_query

from pipeline.config import PreprocConfiguration
from pipeline.config import SciProcConfiguration

from pipeline.services.database import DatabaseHandler
from pipeline.services.blueprint import PreprocessGroup, ScienceGroup

import gc
import traceback

# Run like: python gen_config.py 2>&1 | tee gen_config_tee.log

OVERWRITE_CONFIG = False
OVERWRITE_DATA = False
OVERWRITE_SCIENCE = True
USE_SYSTEM_QUEUE = True

query = """
    SELECT DISTINCT date
    FROM survey_night
    WHERE date >= '2023-10-18'
      AND date < '2027-01-01'
    ORDER BY date;
"""

rows = free_query(query, [])
dates = [r[0].strftime("%Y-%m-%d") for r in rows]  # [::-1]

dbh = DatabaseHandler()

for date in dates:
    try:
        generated_ids = []
        dr = DataReduction([date], use_db=True)

        dr.create_config(overwrite=OVERWRITE_CONFIG)
        for group in dr.blueprint.groups:
            if group.__class__ == PreprocessGroup:
                gen_id = dbh.create_process_data(PreprocConfiguration(group.config))
            elif group.__class__ == ScienceGroup:
                gen_id = dbh.create_process_data(SciProcConfiguration(group.config))
            generated_ids.append(gen_id)
        print(f"Generated {len(set(generated_ids))} process data for {date}\n\n")

    except Exception as e:
        msg = f"Error processing {date}: {e}\n"
        tb = traceback.format_exc()
        print(msg)
        print(tb)
        with open("gen_config_errors.log", "a") as f:
            f.write(msg)
            f.write(tb)
