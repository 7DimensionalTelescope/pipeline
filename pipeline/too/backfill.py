from ..services.database.too import TooDB
from ..services.database.query import RawImageQuery
import datetime
from ..services.blueprint import Blueprint
from ..services.scheduler import Scheduler


def backfill_too(i, overwrite=False, use_system_queue=True, **kwargs):
    too = TooDB()
    dt = too.read_data_by_id(i)

    if dt is not None:
        tile = dt["tile"]
        tile = tile.replace("/", "_")
        trigger_time = dt["trigger_time"]
        image_list = []

        for j in range(100):
            trig = trigger_time.date() + datetime.timedelta(hours=-24 + j)
            trig = trig.strftime("%Y-%m-%d")

            try:
                query_result = RawImageQuery().for_target(tile).on_date(trig).fetch()
                if query_result and "sci" in query_result:
                    image_list = query_result["sci"]

                    if image_list:
                        available_dates = list(set([img["obstime"].date().strftime("%Y%m%d") for img in image_list]))
                        if available_dates:
                            print(trigger_time, image_list[0]["obstime"])
                            break
            except Exception as e:
                print(f"Error querying images for date {trig}: {e}")
                continue

        if not image_list:
            print(f"No images found for TOO {i} (tile: {tile})")
            return

        list_of_images = [data["file_path"] for data in image_list]

        if not list_of_images:
            print(f"No file paths found for TOO {i}")
            return

        try:

            bp = Blueprint.from_list(list_of_images=list_of_images, is_too=True)
            bp.create_config(overwrite=overwrite, is_too=True)
            bp.create_schedule(is_too=True, input_type="ToO")

            if len(bp.schedule) == 0:
                print(f"Empty schedule created for TOO id = {i}")
                return

            sc = Scheduler(
                bp.schedule, use_system_queue=use_system_queue, overwrite=overwrite, overwrite_schedule=True, **kwargs
            )
            if use_system_queue:
                sc.start_system_queue()
            else:
                from ..services.queue import QueueManager

                queue = QueueManager()
                queue.add_scheduler(sc)
                queue.wait_until_task_complete()
            print(f"Successfully added schedule for TOO id = {i} with {len(bp.schedule)} jobs")
        except Exception as e:
            print(f"Error processing TOO {i}: {e}")
            raise
