import os
import shutil
import tempfile

import numpy as np
from astropy.io import fits

from ..config.utils import get_key
from ..utils import atleast_1d


class IntermediateStorageMixin:
    def _prepare_intermediate_storage(self, images):
        if getattr(self, "_intermediate_policy_ready", False):
            return
        requested = self.plan.intermediate_policy
        use_memory = requested == "memory" or (
            requested == "auto" and len(atleast_1d(images)) <= self.plan.memory_image_limit
        )
        durable_models = bool(
            get_key(self.config_node.imcoadd, "output_bkg_map", default=False)
            or get_key(self.config_node.imcoadd, "output_sky_rms_map", default=False)
        )
        if use_memory and durable_models:
            if requested == "memory":
                raise ValueError(
                    "intermediate_policy 'memory' is incompatible with output_bkg_map or output_sky_rms_map"
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
        self._intermediate_policy = "memory" if use_memory else "disk"
        self._frame_cache = {}
        self._working_mask_paths = []
        self._memory_bkgsub_dump_pairs = []
        if use_memory:
            root = tempfile.mkdtemp(prefix="pipeline_imcoadd_", dir="/dev/shm")
            self._memory_intermediate_dir = root
            self._bkgsub_dir = os.path.join(root, "bkgsub")
            self._conv_dir = os.path.join(root, "conv")
            self._weight_dir = os.path.join(root, "weight")
            self._interp_dir = os.path.join(root, "interp")
            os.makedirs(self._bkgsub_dir, exist_ok=True)
            os.makedirs(self._conv_dir, exist_ok=True)
            os.makedirs(self._weight_dir, exist_ok=True)
            os.makedirs(self._interp_dir, exist_ok=True)
        else:
            self._memory_intermediate_dir = None
            self._bkgsub_dir = self.path.imcoadd.factory.bkgsub_dir
            self._conv_dir = self.path.imcoadd.factory.conv_dir
            self._weight_dir = self.path.imcoadd.factory.weight_dir
            self._interp_dir = self.path.imcoadd.factory.interp_dir
        self._intermediate_policy_ready = True
        self.logger.info(
            f"Intermediate policy: {self._intermediate_policy} "
            f"({len(atleast_1d(images))} images, memory limit {self.plan.memory_image_limit})"
        )

    def _read_stage_frame(self, image):
        cached = getattr(self, "_frame_cache", {}).get(image)
        if cached is not None:
            return cached
        data, header = fits.getdata(image, header=True, memmap=False)
        value = np.ascontiguousarray(data, dtype=np.float32), header
        if getattr(self, "_intermediate_policy", "disk") == "memory":
            self._frame_cache[image] = value
        return value

    def _store_stage_frame(self, image, data, header):
        value = np.ascontiguousarray(data, dtype=np.float32), header.copy()
        if getattr(self, "_intermediate_policy", "disk") == "memory":
            self._frame_cache[image] = value
            if self.plan.mode == "proper":
                fits.writeto(image, value[0], header=value[1], overwrite=True)
        else:
            fits.writeto(image, value[0], header=value[1], overwrite=True)

    def _stage_frame_exists(self, image):
        return image in getattr(self, "_frame_cache", {}) or os.path.exists(image)

    def discard_cached_frames(self):
        getattr(self, "_frame_cache", {}).clear()

    def _cleanup_imcoadd_intermediates(self):
        root = getattr(self, "_memory_intermediate_dir", None)
        plan = getattr(self, "_plan", None)
        if root and plan is not None and plan.dump_bkgsub and getattr(self, "_coadd_completed", False):
            for source, destination in self._memory_bkgsub_dump_pairs:
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                cached = self._frame_cache.get(source)
                if cached is not None:
                    fits.writeto(destination, cached[0], header=cached[1], overwrite=True)
                elif os.path.exists(source):
                    shutil.copy2(source, destination)
        if root:
            shutil.rmtree(root, ignore_errors=True)
            destinations = [destination for _, destination in self._memory_bkgsub_dump_pairs]
            self.config_node.imcoadd.bkgsub_images = (
                destinations if plan is not None and plan.dump_bkgsub else None
            )
            self.config_node.imcoadd.conv_files = None
            self.images_to_coadd = None
        for path in getattr(self, "_working_mask_paths", []):
            try:
                os.remove(path)
            except OSError:
                pass
        self.discard_cached_frames()
        self._intermediate_policy_ready = False
