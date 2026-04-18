import pandas as pd
from pipeline.services.scheduler import Scheduler
from pipeline.services.database import DatabaseHandler

"""Run like: python run_reprocess_preprocess.py 2>&1 | tee 2026-04-18_reprocess_preprocess_tee.log"""

BASE_PRIORITY = 1
OVERWRITE_CONFIG = False
OVERWRITE_DATA = False

dh = DatabaseHandler()
df = dh.process_status.export_to_table()

preproc_df = df[df["config_type"] == "preprocess"]
df_errors = preproc_df[~pd.isna(preproc_df["errors"])]
df_rerun = df_errors[df_errors["errors"] == 102]

config_files = df_rerun["config_file"].tolist()
nightdates = [date.strftime("%Y-%m-%d") for date in set(df_rerun["nightdate"])]
print(f"Processing {len(config_files)} config files: nightdates {nightdates}")

sc = Scheduler.from_list(
    config_files,
    overwrite_config=OVERWRITE_CONFIG,
    overwrite_data=OVERWRITE_DATA,
    use_system_queue=True,
    base_priority=BASE_PRIORITY,
)
sc.start_system_queue()
