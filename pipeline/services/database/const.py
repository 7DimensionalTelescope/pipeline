import os
import warnings

# relies on const.environ loading dotenv first, which it always does.
dbname = os.environ.get("DBNAME")
user = os.environ.get("DBUSER")
host = os.environ.get("DBHOST")
port = os.environ.get("DBPORT")
password = os.environ.get("DBPASSWORD")


def _remote_from_env() -> dict:
    """Remote profile, re-read from the environment on every call."""

    return {
        "dbname": os.environ.get("REMOTE_DBNAME", dbname),
        "user": os.environ.get("REMOTE_DBUSER", user),
        "host": os.environ.get("REMOTE_DBHOST"),
        "port": os.environ.get("REMOTE_DBPORT", port),
        "password": os.environ.get("REMOTE_DBPASSWORD", password),
    }


# Connection profiles. "local" is the shipping default and is exactly the
# historical DB_PARAMS. "remote" reaches the same Postgres from another host
# over TCP, taking REMOTE_* from .env and falling back to the local value for
# anything unset (REMOTE_DBHOST is mandatory - see _resolve_profile).
DB_PROFILES = {
    "local": {
        "dbname": dbname,
        "user": user,
        "host": host,
        "port": port,
        "password": password,
    },
    "remote": _remote_from_env(),
}


def _resolve_profile(backend: str) -> dict:
    """Validated profile lookup; keywords: backend name, remote host required."""

    key = str(backend).strip().lower()
    if key not in DB_PROFILES:
        raise ValueError(f"Unknown DB backend {backend!r}. Choose from {sorted(DB_PROFILES)}.")
    if key == "remote":
        DB_PROFILES["remote"] = _remote_from_env()  # honour REMOTE_* set after import
    params = DB_PROFILES[key]
    if key == "remote" and not params["host"]:
        raise ValueError("DB backend 'remote' requires REMOTE_DBHOST in .env (would otherwise connect locally).")
    return dict(params)


_env_backend = os.environ.get("DB_BACKEND", "local")
try:
    _initial = _resolve_profile(_env_backend)
except ValueError as exc:
    warnings.warn(f"{exc} Falling back to the 'local' DB backend.", stacklevel=2)
    _env_backend, _initial = "local", _resolve_profile("local")

DB_BACKEND = _env_backend.strip().lower()

# Mutated in place by set_db_backend: base.py, query.py and gwportal.py all
# hold a reference to this very dict, so rebinding it would not reach them.
DB_PARAMS = _initial


def set_db_backend(backend: str) -> dict:
    """Switch every Postgres consumer to a connection profile; keywords: local, remote."""

    global DB_BACKEND

    params = _resolve_profile(backend)
    DB_PARAMS.clear()
    DB_PARAMS.update(params)
    DB_BACKEND = str(backend).strip().lower()

    from .query import close_pool  # late: query.py imports from this module

    close_pool()
    return describe_db_backend()


def describe_db_backend() -> dict:
    """Active backend and its parameters, password masked; safe to print or log."""

    described = {"backend": DB_BACKEND, **DB_PARAMS}
    if described.get("password"):
        described["password"] = "***"
    return described


def get_db_backend() -> str:
    """Name of the active connection profile."""

    return DB_BACKEND


# GWPortal REST API credentials (used by the HTTP backend)
GWPORTAL_BASE_URL = os.environ.get("GWPORTAL_BASE_URL")
GWPORTAL_API_KEY = os.environ.get("GWPORTAL_API_KEY")


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
