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


def sync_images(images) -> dict:
    """Sync a list of images, the sciproc configs they belong to, and the config-level edges.

    The config goes first so its `process_status.id` can be stamped on every `image_qa` row
    it owns; the image-level dependency graph is then rolled up per config. One bad image or
    a config that is not on disk is recorded in `errors` and never aborts the rest.

    Returns {"configs": {yml: {...}}, "images": {path: {...}}, "errors": {path: repr}}.
    """
    from ...path import PathHandler
    from ...utils import atleast_1d
    from .process_status_dependency import ProcessStatusDependency

    images = [os.path.abspath(image) for image in atleast_1d(images) if image]
    report = {"configs": {}, "images": {}, "errors": {}}
    if not images:
        return report

    def resolve(paths):
        """Config per path. PathHandler reads _output_dir, so this does not mkdir; a uniform
        group collapses to one string, hence the broadcast."""
        ymls = PathHandler(paths, is_pipeline=True).sciproc_output_yml
        return list(ymls) if isinstance(ymls, (list, tuple)) else [ymls] * len(paths)

    try:
        ymls = resolve(images)
    except Exception:
        # one unparseable name raises for the whole batch -- fall back so it costs only itself
        ymls = []
        for image in images:
            try:
                ymls.append(resolve([image])[0])
            except Exception as exc:
                report["errors"][image] = repr(exc)
                ymls.append(None)  # config unknown; the image is still registered below

    by_config = {}
    for image, yml in zip(images, ymls):
        by_config.setdefault(yml, []).append(image)

    for yml, group in by_config.items():
        ps_id = status = None
        if yml is not None:
            try:
                ps_id, status = sync_config(yml)
            except Exception as exc:  # config not on disk, unreadable flags, DB error
                report["errors"][yml] = repr(exc)
            report["configs"][yml] = {"process_status_id": ps_id, "status": status, "n_images": len(group)}

        synced = 0
        for image in group:
            try:
                qa_id, n_dep = sync_image(image, ps_id)
                report["images"][image] = {"qa_id": qa_id, "n_dependency_rows": n_dep}
                synced += 1
            except Exception as exc:
                report["errors"][image] = repr(exc)

        if yml is not None and ps_id is not None and synced:
            report["configs"][yml]["n_rollup_edges"] = ProcessStatusDependency().sync_from_products(ps_id)

    return report


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
