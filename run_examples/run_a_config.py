from pipeline.services.scheduler import Scheduler

f = "/lyman/data2/processed/2025-12-06/2025-12-06_7DT04.yml"
sc = Scheduler.from_list([f])
sc.start_system_queue()
