"""
Human inspection verdicts: FITS headers, image_qa, and process_status in one call.

A verdict is `SANITY` plus an `INSPCOMM` comment. `INSPCOMM` seals `SANITY` against
recomputation (see `Checker._sanity_action`), so this is the one authorized way to write
it: the header is the truth, `image_qa.sanity`/`inspectd` is its index, and
`process_status.sanity` is the config-level decision automation reads
(`WHERE sanity IS NOT FALSE`).

Idempotent: repeating a call writes nothing. INSPCOMM is freeform, so an existing comment
is left exactly as it is; pass `overwrite=True` to replace it with the new one.

    from pipeline.services.inspection import inspect_config
    inspect_config("/lyman/data2/processed/2024-01-26/UDS/m550/UDS_m550_2024-01-26.yml",
                   sanity=False, inspcomm="Mount trailed; bad flat")
"""

import os
from typing import List, Optional, Union

from astropy.io import fits

from ..utils.header import update_padded_header_smart
from ..utils import atleast_1d

INSPCOMM_KEY = "INSPCOMM"
INSPCOMM_COMMENT = "Human inspection comment; Trust SANITY if exists"
SANITY_COMMENT = "Pipeline image sanity flag"


def resolve_inspcomm(existing: Optional[str], new: Optional[str], overwrite: bool = False) -> Optional[str]:
    """The INSPCOMM to store: keep `existing` unless it is empty or `overwrite`. No `new` never erases."""
    if not new or (existing and not overwrite):
        return existing
    return new


def inspect_images(
    images: Union[str, List[str]],
    sanity: bool,
    inspcomm: str = None,
    by: str = None,
    overwrite: bool = False,
    use_database: bool = True,
) -> dict:
    """Seal a human verdict into image headers and mirror it into image_qa."""
    images = [image for image in atleast_1d(images) if image]
    report = {"headers_written": [], "headers_unchanged": [], "image_qa_updated": [], "image_qa_missing": []}

    image_qa = None
    if use_database:
        from .database.image_qa import ImageQA

        image_qa = ImageQA()

    for image in images:
        header = fits.getheader(image)
        resolved = resolve_inspcomm(header.get(INSPCOMM_KEY), inspcomm, overwrite)

        if header.get("SANITY") is sanity and resolved == header.get(INSPCOMM_KEY):
            report["headers_unchanged"].append(image)
        else:
            cards = {"SANITY": (sanity, SANITY_COMMENT)}
            if resolved is not None:
                cards[INSPCOMM_KEY] = (resolved, INSPCOMM_COMMENT)
            update_padded_header_smart(image, cards)
            report["headers_written"].append(image)

        if image_qa is None:
            continue

        image_name = os.path.basename(image).replace(".fits", "")
        qa_id = image_qa.read_data_by_params(image_name=image_name)
        if qa_id is None:
            report["image_qa_missing"].append(image_name)
            continue

        for one_id in atleast_1d(qa_id):
            image_qa.update_data(int(one_id), sanity=sanity, inspectd=True)
            report["image_qa_updated"].append(int(one_id))

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
