"""Post-hoc DB sync: image_qa upsert + dependency rows for images, and process_status
reconciliation from config flags. Backs the ``db_sync`` CLI; importable directly."""

import os

from ...const.sciproc import SCIPROCESS_REGISTRY

# status strings the pipeline itself records when each stage completes
COMPLETED_STATUS = {
    "astrometry": "astrometry",
    "single_photometry": "single_photometry-completed",
    "coadd": "imcoadd-completed",
    "coadd_photometry": "coadd_photometry-completed",
    "subtraction": "imsubtract-completed",
    "difference_photometry": "difference_photometry-completed",
}


def sync_image(path: str, process_status_id: int | None = None) -> tuple[int, int]:
    """Upsert image_qa from the on-disk header, then rebuild its dependency rows.

    Returns (qa_id, n_dependency_rows).
    """
    from .image_qa import ImageQA, ImageQATable
    from .image_qa_dependency import ImageQADependency

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    qa_id = ImageQA().create_data(ImageQATable.from_file(path, process_status_id))
    n_dep = ImageQADependency().sync(path, qa_id)
    return qa_id, n_dep


def last_completed_spec(config_file: str):
    """Furthest spec in chain order whose flag is set, or None if nothing completed."""
    from ...config import SciProcConfiguration

    flags = SciProcConfiguration(config_file, write=False, logger=False).node.flag
    done = None
    for spec in SCIPROCESS_REGISTRY.specs:
        if not getattr(flags, spec.name, False):
            break
        done = spec
    return done


def sync_config(config_file: str) -> tuple[int | None, str | None]:
    """Reconcile the process_status row (matched on config_file, newest if several)
    with the config's flags; creates the row if none exists.

    Returns (process_status_id, status) -- status None when no stage has completed.
    """
    from .process_status import ProcessStatus
    from .query import free_query

    config_file = os.path.abspath(config_file)
    if not os.path.exists(config_file):
        raise FileNotFoundError(config_file)
    ps = ProcessStatus()
    rows = free_query("SELECT id FROM process_status WHERE config_file = %s ORDER BY id DESC LIMIT 1", (config_file,))
    ps_id = rows[0][0] if rows else ps.create_data(config_file)

    spec = last_completed_spec(config_file)
    if spec is None:
        return ps_id, None
    ps.update_data(ps_id, progress=spec.progress_end, status=COMPLETED_STATUS[spec.name])
    return ps_id, COMPLETED_STATUS[spec.name]
