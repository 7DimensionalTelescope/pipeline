import os
import time
from astropy.io import fits

from ..utils import add_suffix, get_basename, time_diff_in_seconds
from ..path.path import PathHandler

from .interpolate import interpolate_masked_pixels, add_bpx_method


def apply_bpmask(self, badpix=0, device_id=None, use_gpu: bool = True):
    st = time.time()
    self._use_gpu = all([use_gpu, self.config.imstack.gpu, self._use_gpu])

    # if self._use_gpu:
    #     device_id = self.get_device_id(device_id)
    # else:
    #     device_id = "CPU"

    device_id = 1

    st = time.time()
    self.logger.info("Start the interpolation for bad pixels")

    path_interp = os.path.join(self.path_tmp, "interp")
    os.makedirs(path_interp, exist_ok=True)

    self.config.imstack.interp_images = [
        os.path.join(path_interp, add_suffix(get_basename(f), "interp")) for f in self.input_images
    ]

    # bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)
    # ad-hoc. Todo: group input_files for different bpmask files
    bpmask_array, header = fits.getdata(PathHandler.get_bpmask(self.config.imstack.bkgsub_images[0]), header=True)

    if "BADPIX" in header.keys():
        badpix = header["BADPIX"]
        self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
    else:
        self.logger.warning("BADPIX not found in header. Using default value 0.")

    # pre-bind to avoid UnboundLocalError
    # interp = image = weight = interp_weight = None

    method = self.config.imstack.interp_type
    _calc_weight = self.config.imstack.weight_map

    uncalculated_images = []
    for i in range(len(self.config.imstack.bkgsub_images)):
        input_image_file = self.config.imstack.bkgsub_images[i]
        output_file = self.config.imstack.interp_images[i]

        if os.path.exists(output_file) and not self.overwrite:
            self.logger.debug(f"Already exists; skip generating {output_file}")
            continue
        else:
            if _calc_weight:
                uncalculated_images.append([input_image_file, input_weight_file, output_file, output])
            else:
                uncalculated_images.append(input_image_file)

    output, output_weight = interpolate_masked_pixels(
        uncalculated_images, bpmask_array, method=method, badpix=badpix, device=device_id, weight=weight
    )

    for i, files in enumerate(uncalculated_images):
        input_image_file, input_weight_file, output_file, output = files
        if _calc_weight:
            input_weight_file = add_suffix(input_image_file, "weight")
            output_weight_file = add_suffix(output_file, "weight")
            fits.writeto(
                output_weight_file,
                data=output_weight[i],
                header=add_bpx_method(fits.getheader(input_weight_file), method),
                overwrite=True,
            )
        fits.writeto(
            output_file,
            data=output[i],
            header=add_bpx_method(fits.getheader(input_image_file), method),
            overwrite=True,
        )

    self.images_to_stack = self.config.imstack.interp_images

    for attr in ("output", "output_weight"):
        if attr in self.__dict__:
            del self.__dict__[attr]

    self.logger.info(
        f"Interpolation for bad pixels is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(self.images_to_stack):.1f} s/image)"
    )
