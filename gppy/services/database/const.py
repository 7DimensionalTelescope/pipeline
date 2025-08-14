import os
from dotenv import load_dotenv

env_file = os.path.join(os.path.dirname(__file__), "../../../.env")
load_dotenv(env_file)

dbname = os.environ["DBNAME"]
user = os.environ["DBUSER"]
host = os.environ["DBHOST"]
port = os.environ["DBPORT"]
password = os.environ["DBPASSWORD"]

DB_PARAMS = {
    "dbname": dbname,
    "user": user,
    "host": host,
    "port": port,
    "password": password,
}


TABLES = {
    "sci": "survey_scienceframe",
    "bias": "survey_biasframe",
    "dark": "survey_darkframe",
    "flat": "survey_flatframe",
}

ALIASES = {
    "sci": "sci",
    "science": "sci",
    "scienceframe": "sci",
    "bias": "bias",
    "biasframe": "bias",
    "dark": "dark",
    "darkframe": "dark",
    "flat": "flat",
    "flatframe": "flat",
}
