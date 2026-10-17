"""
Human inspection verdicts: image_qa, FITS headers, and process_status in one call.

A verdict is `SANITY` plus an `INSPCOMM` comment. `INSPCOMM` beside `SANITY=False` is a
veto: final, never recomputed (see `Checker._sanity_action`). `INSPCOMM` beside
`SANITY=True` is a soft record: the machine recomputes it like any other frame and, when
it rejects, `machine_reject` removes the comment. This module is the one authorized way to
write a verdict: `image_qa.inspcomm` is the durable copy, which `Preprocess.prepare_header`
restores into a rebuilt header, the FITS header and its `.header` sidecar carry the same
cards, and `process_status.sanity` is the config-level decision automation reads.

Idempotent: repeating a call writes nothing. INSPCOMM is freeform, so an existing comment
is left exactly as it is; pass `overwrite=True` to replace it with the new one.

    from pipeline.services.inspection import inspect_config
    inspect_config("/lyman/data2/processed/2024-01-26/UDS/m550/UDS_m550_2024-01-26.yml",
                   sanity=False, inspcomm="Mount trailed; bad flat")
"""

import os
import warnings
from typing import List, Optional, Union

from astropy.io import fits

from ..const.sciproc import REJECTION_PROCESS_HEADER_KEY
from ..utils.header import (
    read_header_file,
    remove_padded_header_key,
    update_padded_header_cards,
    update_padded_header_file,
    update_padded_header_smart,
    write_header_file,
)
from ..utils import atleast_1d, swap_ext

INSPCOMM_KEY = "INSPCOMM"
INSPCOMM_COMMENT = "Human inspection; trust SANITY=F"
SANITY_COMMENT = "Deliverable per machine QA; INSPCOMM+F = veto"
REJ_PROC_COMMENT = "Sci-process that set SANITY to False"


def resolve_inspcomm(existing: Optional[str], new: Optional[str], overwrite: bool = False) -> Optional[str]:
    """The INSPCOMM to store: keep `existing` unless it is empty or `overwrite`. No `new` never erases."""
    if not new or (existing and not overwrite):
        return existing
    return new


def machine_reject(image: str, rej_proc: Optional[str], cards: Union[dict, fits.Header, None] = None) -> bool:
    """One FITS write: the caller's `cards`, then SANITY=F (+REJ_PROC); removes a soft INSPCOMM, never a veto. Returns whether one was removed."""
    seal = {"SANITY": (False, SANITY_COMMENT)}
    if rej_proc:
        seal[REJECTION_PROCESS_HEADER_KEY] = (rej_proc, REJ_PROC_COMMENT)
    with fits.open(image, mode="update") as hdul:
        header = hdul[0].header
        if INSPCOMM_KEY in header and header.get("SANITY") is False:
            return False  # a veto is final; the chain's inspection filter keeps vetoes away from every caller
        removed = remove_padded_header_key(header, INSPCOMM_KEY)
        if cards is not None:
            update_padded_header_cards(header, cards)
        update_padded_header_cards(header, seal)
        hdul.flush()

    if removed:  # the sidecar is the header's twin: no INSPCOMM here means none there
        header_file = swap_ext(image, "header")
        if os.path.exists(header_file):
            sidecar = read_header_file(header_file)
            if sidecar.get("SANITY") is not False and remove_padded_header_key(sidecar, INSPCOMM_KEY):
                write_header_file(header_file, sidecar)
    return removed


def inspect_images(
    images: Union[str, List[str]],
    sanity: bool,
    inspcomm: str = None,
    by: str = None,
    overwrite: bool = False,
    use_database: bool = True,
) -> dict:
    """Seal a human verdict: image_qa first (the durable copy), then the FITS header and its sidecar."""
    images = [image for image in atleast_1d(images) if image]
    report = {
        "headers_written": [],
        "headers_unchanged": [],
        "sidecars_written": [],
        "sidecars_unchanged": [],
        "sidecars_missing": [],
        "image_qa_updated": [],
        "image_qa_registered": [],
        "image_qa_missing": [],
    }

    image_qa = None
    if use_database:
        from .database.image_qa import ImageQA

        image_qa = ImageQA()
    else:
        warnings.warn("use_database=False: the verdict is not durable across a Preprocess rebuild")

    for image in images:
        header = fits.getheader(image)
        resolved = resolve_inspcomm(header.get(INSPCOMM_KEY), inspcomm, overwrite)
        if not resolved:
            raise ValueError(f"{os.path.basename(image)}: an INSPCOMM comment is required; it is what seals SANITY")

        if image_qa is not None:
            image_name = os.path.basename(image).replace(".fits", "")
            if image_qa.read_data_by_params(image_name=image_name) is None:
                from .database.sync import sync_images

                # singles get a row only from Astrometry, after the inspection filter: a frame rejected first has none
                if image in sync_images([image])["images"]:
                    report["image_qa_registered"].append(image)
                else:
                    report["image_qa_missing"].append(image_name)
            stored = image_qa.set_inspection(image_name, sanity, resolved, overwrite=overwrite, image_path=image)
            report["image_qa_updated"] += list(stored)
            resolved = next(iter(stored.values()), resolved)

        cards = {"SANITY": (sanity, SANITY_COMMENT), INSPCOMM_KEY: (resolved, INSPCOMM_COMMENT)}
        if header.get("SANITY") is sanity and resolved == header.get(INSPCOMM_KEY):
            report["headers_unchanged"].append(image)
        else:
            update_padded_header_smart(image, cards)
            report["headers_written"].append(image)

        # the sidecar is judged on its own: the FITS may already carry the verdict from an earlier hand injection
        report[f"sidecars_{update_padded_header_file(image, cards)}"].append(image)

    return report


def inspect_config(
    config,
    sanity: bool,
    inspcomm: str = None,
    by: str = None,
    overwrite: bool = False,
    images: bool = True,
    use_database: bool = True,
) -> dict:
    """
    Seal a human verdict on a whole config: its input images and its process_status row.

    `images=False` records the config-level verdict only.
    """
    from ..config import SciProcConfiguration

    if isinstance(config, str):
        config = SciProcConfiguration(config, write=False, logger=False)

    report = {"config": config.node.name}

    if images:
        input_images = config.node.input.calibrated_images
        if not input_images:
            report["images"] = "no calibrated_images in config"
        else:
            report["images"] = inspect_images(
                input_images, sanity=sanity, inspcomm=inspcomm, by=by, overwrite=overwrite, use_database=use_database
            )

    if use_database:
        from .database.process_status import ProcessStatus

        report["process_status_id"] = ProcessStatus().set_inspection(
            config.node.name, sanity=sanity, inspcomm=inspcomm, by=by, overwrite=overwrite
        )

    return report
