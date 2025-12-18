from ..services.database.too import TooDB
from ..services.database.query import RawImageQuery
import datetime
from ..services.blueprint import Blueprint
from ..services.scheduler import Scheduler
import os
from ..run import query_observations


def backfill_too(i):
    too = TooDB()
    dt = too.read_too_data_by_id(i)

    if dt is not None:
        tile = dt["tile"]
        tile = tile.replace("/", "_")
        trigger_time = dt["trigger_time"]
        for j in range(10):
            trig = trigger_time.date() + datetime.timedelta(days=-1 + j)
            trig = trig.strftime("%Y-%m-%d")

            image_list = RawImageQuery().for_target(tile).on_date(trig).fetch()["sci"]

            available_dates = list(set([img["obstime"].date().strftime("%Y%m%d") for img in image_list]))
            if available_dates:
                print(trigger_time, image_list[0]["obstime"])
                break

        list_of_images = [data["file_path"] for data in image_list]

        br = Blueprint.from_list(list_of_images=list_of_images, is_too=True)
        br.create_config(overwrite=True, is_too=True)
        br.create_schedule(is_too=True)
        sc = Scheduler(br.schedule, use_system_queue=True)
        sc.start_system_queue()
