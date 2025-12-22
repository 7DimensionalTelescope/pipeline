from ..services.database.too import TooDB
from ..services.database.query import RawImageQuery
import datetime
from ..services.blueprint import Blueprint
from ..services.scheduler import Scheduler


def backfill_too(i, overwrite=False, **kwargs):
    too = TooDB()
    dt = too.read_too_data_by_id(i)

    if dt is not None:
        tile = dt["tile"]
        tile = tile.replace("/", "_")
        trigger_time = dt["trigger_time"]
        image_list = []

        for j in range(10):
            trig = trigger_time.date() + datetime.timedelta(days=-1 + j)
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
            br = Blueprint.from_list(list_of_images=list_of_images, is_too=True)
            br.create_config(overwrite=overwrite, is_too=True)
            br.create_schedule(is_too=True)

            if len(br.schedule) == 0:
                print(f"Empty schedule created for TOO {i}")
                return

            sc = Scheduler(br.schedule, use_system_queue=True, overwrite=overwrite, **kwargs)
            sc.start_system_queue()
            print(f"Successfully added schedule for TOO {i} with {len(br.schedule)} jobs")
        except Exception as e:
            print(f"Error processing TOO {i}: {e}")
            raise
