import os

CalibType = ["BIAS", "DARK", "FLAT"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.environ["REF_DIR"]

RAWDATA_DIR = os.environ["RAWDATA_DIR"]
PROCESSED_DIR = os.environ["PROCESSED_DIR"]
MASTER_FRAME_DIR = os.environ["MASTER_FRAME_DIR"]

FACTORY_DIR = os.environ["FACTORY_DIR"]
SLACK_TOKEN = os.environ["SLACK_TOKEN"]

available_7dt_units = [f"7DT0{unit}" if unit < 10 else f"7DT{unit}" for unit in range(1, 20)]

class PipelineError(Exception):
    pass
