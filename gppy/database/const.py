import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

dbname = os.environ["DBNAME"]
user = os.environ["DBUSER"]
host = os.environ["DBHOST"]
port = os.environ["DBPORT"]
password = os.environ["DBPASSWORD"]
