import os
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

from ..config.utils import get_key
from ..path.path import PathHandler
from ..services.logger import Logger
from ..utils import atleast_1d
from .coadd_plan import CoaddPlan
from .counts import CoaddCounts


@dataclass(slots=True)
class IntermediateStorage:
    policy: str
    root: str | None
    bkgsub_dir: str
    conv_dir: str
    weight_dir: str
    interp_dir: str
    source_mask_dir: str
    frame_cache: dict[str, tuple[np.ndarray, fits.Header]]
    working_mask_paths: list[str]
    bkgsub_dump_pairs: list[tuple[str, str]]


class IntermediateStorageMixin:
    logger: Logger
    path: PathHandler
    plan: CoaddPlan
    intermediate_storage: IntermediateStorage | None

    @property
    def storage(self) -> IntermediateStorage:
        if self.intermediate_storage is None:
            raise RuntimeError("Intermediate storage has not been prepared")
        return self.intermediate_storage

    def _prepare_intermediate_storage(self, images):
        requested = self.plan.intermediate_policy
        use_memory = requested == "memory" or (
            requested == "auto" and len(atleast_1d(images)) <= self.plan.memory_image_limit
        )
        durable_models = self.plan.output_bkg_map or self.plan.output_sky_rms_map
        if use_memory and durable_models:
            if requested == "memory":
                raise ValueError(
                    "intermediate_policy 'memory' is incompatible with output_bkg_map or output_sky_rms_map"
                )
            use_memory = False
        if use_memory and self.plan.routine == "legacy":
            if requested == "memory":
                raise ValueError(
                    "intermediate_policy 'memory' is incompatible with coadd_routine 'legacy' (SWarp reads files)"
                )
            use_memory = False
        if use_memory:
            need = int(4.5 * sum(os.path.getsize(path) for path in atleast_1d(images)))
            usable = os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK)
            free = shutil.disk_usage("/dev/shm").free if usable else 0
            if not usable or free < need:
                if requested == "memory":
                    raise OSError(
                        f"intermediate_policy 'memory' needs {need / 1e9:.1f} GB in /dev/shm; "
                        f"{free / 1e9:.1f} GB is available"
                    )
                use_memory = False
        policy = "memory" if use_memory else "disk"
        if use_memory:
            root = tempfile.mkdtemp(prefix="pipeline_imcoadd_", dir="/dev/shm")
            bkgsub_dir = os.path.join(root, "bkgsub")
            conv_dir = os.path.join(root, "conv")
            weight_dir = os.path.join(root, "weight")
            interp_dir = os.path.join(root, "interp")
            source_mask_dir = os.path.join(root, "srcmask")
            os.makedirs(bkgsub_dir, exist_ok=True)
            os.makedirs(conv_dir, exist_ok=True)
            os.makedirs(weight_dir, exist_ok=True)
            os.makedirs(interp_dir, exist_ok=True)
        else:
            factory = self.path.imcoadd.factory
            root = None
            bkgsub_dir = factory.bkgsub_dir
            conv_dir = factory.conv_dir
            weight_dir = factory.weight_dir
            interp_dir = factory.interp_dir
            source_mask_dir = factory.source_mask_dir
        self.intermediate_storage = IntermediateStorage(
            policy=policy,
            root=root,
            bkgsub_dir=bkgsub_dir,
            conv_dir=conv_dir,
            weight_dir=weight_dir,
            interp_dir=interp_dir,
            source_mask_dir=source_mask_dir,
            frame_cache={},
            working_mask_paths=[],
            bkgsub_dump_pairs=[],
        )
        self.logger.info(
            f"Intermediate policy: {policy} "
            f"({len(atleast_1d(images))} images, memory limit {self.plan.memory_image_limit})"
        )

    def _read_stage_frame(self, image):
        storage = self.storage
        cached = storage.frame_cache.get(image)
        if cached is not None:
            return cached
        data, header = fits.getdata(image, header=True, memmap=False)
        value = np.ascontiguousarray(data, dtype=np.float32), header
        if storage.policy == "memory":
            storage.frame_cache[image] = value
        return value

    def _store_stage_frame(self, image, data, header):
        storage = self.storage
        value = np.ascontiguousarray(data, dtype=np.float32), header.copy()
        if storage.policy == "memory":
            storage.frame_cache[image] = value
            if self.plan.mode == "proper":
                fits.writeto(image, value[0], header=value[1], overwrite=True)
        else:
            fits.writeto(image, value[0], header=value[1], overwrite=True)

    def _stage_frame_exists(self, image):
        return image in self.storage.frame_cache or os.path.exists(image)

    def discard_cached_frames(self):
        self.storage.frame_cache.clear()

    def _cleanup_imcoadd_intermediates(self):
        storage = self.intermediate_storage
        if storage is None:
            return
        root = storage.root
        dumped = root and self.plan.dump_bkgsub and self._coadd_completed
        if dumped:
            for source, destination in storage.bkgsub_dump_pairs:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                cached = storage.frame_cache.get(source)
                if cached is not None:
                    fits.writeto(destination, cached[0], header=cached[1], overwrite=True)
                elif os.path.exists(source):
                    shutil.copy2(source, destination)
        if root:
            shutil.rmtree(root, ignore_errors=True)
            destinations = [destination for _, destination in storage.bkgsub_dump_pairs]
            self.config_node.imcoadd.bkgsub_images = destinations if dumped else None
            self.config_node.imcoadd.conv_files = None
            self.images_to_coadd = None
            interp_images = atleast_1d(get_key(self.config_node.imcoadd, "interp_images") or [])
            if any(str(path).startswith(root) for path in interp_images):
                self.config_node.imcoadd.interp_images = None
            weight_images = atleast_1d(get_key(self.config_node.imcoadd, "bkgsub_weight_images") or [])
            if any(str(path).startswith(root) for path in weight_images):
                self.config_node.imcoadd.bkgsub_weight_images = None
        for path in storage.working_mask_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        storage.frame_cache.clear()
        # the count planes and the per-frame bit masks are grid-sized and already written by now
        self._coadd_counts = CoaddCounts()
        self._coadd_mask_builder = None
        self._quality_masks = None
        self._badpix_positions_cache = {}
        self._saturated_positions_cache = {}
        self._saturation_map_cache = {}
        self._bpmask_coords_cache = {}
        self.intermediate_storage = None
