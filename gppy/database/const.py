import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

dbname = os.environ["DBNAME"]
user = os.environ["DBUSER"]
host = os.environ["DBHOST"]
port = os.environ["DBPORT"]
password = os.environ["DBPASSWORD"]

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
