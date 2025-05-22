import os
import glob
import threading    
import numpy as np
from astropy.io import fits
import time
import gc

import cupy as cp
from .plotting import *
from . import utils as prep_utils
from .cupy_calc import *

from ..services.queue import QueueManager, Priority
from ..services.memory import MemoryMonitor
from ..utils import add_padding
from ..path import PathHandler
from ..config import PreprocConfiguration
from ..services.setup import BaseSetup

class Preprocess(BaseSetup):
    """
    Assumes homogeneous BIAS, DARK, FLAT, SCI frames as input
    taken on the same date with the same
    unit, n_binning and gain have all identical cameras.
    """

    def __init__(
        self,
        input,
        queue=False,
        logger=None,
        overwrite=False,
        **kwargs,
    ):

        if isinstance(input, PreprocConfiguration):
            self.config = input
        elif isinstance(input, str) or isinstance(input, list):
            self.config = PreprocConfiguration(input)
        else:
            raise ValueError("Input must be a PreProcConfiguration object or a string/list of file paths")

        # Load Configuration
        super().__init__(self.config, logger, queue)

        self.overwrite = overwrite

        # self.logger.debug(f"Masterframe output folder: {self.path_fdz}")

    @classmethod
    def from_list(cls):
        pass

    @property
    def sequential_task(self):
        tasks = [
            (1, "initialize", False),
        ]
        for i in range(self._n_groups):
            tasks.append((4*i+2, f"generate_masterframe", True))
            tasks.append((4*i+3, f"data_reduction", True))
            tasks.append((4*i+4, f"make_plots", False))
            if i < self._n_groups - 1:
                tasks.append((4*i+5, f"proceed_to_next_group", False))
        
        return tasks

    def initialize(self):
        self.logger.info("Initializing Preprocess")
        if self.config.input.masterframe_files and self.config.input.science_files:
            input_files = list(self.config.input.masterframe_files) + list(self.config.input.science_files)
            self.raw_groups = PathHandler.take_raw_inventory(input_files)
        else:
            input_files = glob.glob(os.path.join(self.config.input.raw_dir, "*.fits"))
            self.raw_groups = PathHandler.take_raw_inventory(input_files)

        self._n_groups = len(self.raw_groups)
        self._current_group = 0

        self.choose_device()
        
        self.logger.info(f"{self._n_groups} groups are found")

    def run(self):
        self.initialize()
        threads_for_making_plots = []
        for i in range(self._n_groups):
            self.generate_masterframe()
            self.data_reduction()

            t = threading.Thread(target=self.make_plots, args=(i,))
            t.start()
            threads_for_making_plots.append(t)
            
            if i < self._n_groups - 1:
                self.proceed_to_next_group()

        for t in threads_for_making_plots:
            t.join()

    def choose_device(self):
        if self.config.preprocess.device is None:
            from ..services.memory import MemoryMonitor
            self.logger.debug("No device specified, choosing the device with least memory usage")
            self.config.preprocess.device = np.argmin(MemoryMonitor.current_gpu_memory_percent)
            self.logger.debug(f"Chosen device: {self.config.preprocess.device}")
        self.device_id = self.config.preprocess.device

    def proceed_to_next_group(self):
        self._current_group += 1
        if self._current_group >= self._n_groups:
            raise StopIteration

    def proceed_to_previous_group(self):
        self._current_group -= 1
        if self._current_group < 0:
            raise StopIteration

    def switch_to_group(self, group_index):
        if group_index < 0 or group_index >= self._n_groups:
            raise StopIteration
        self._current_group = group_index

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]

        return self._get_raw_group(name, self._current_group)

    def _get_raw_group(self, name, group_index):
        if name == "sci_input":
            return self._parse_sci_list(group_index, "input")
        elif name == "sci_output":
            return self._parse_sci_list(group_index, "output")
        elif name == "bpmask_output":
            return getattr(self, f"dark_output").replace("dark", f"bpmask")

        key_to_index = {"bias": 0, "dark": 1, "flat": 2}
        if name.endswith("_input"):
            key = name[:4]  # strip "_input" (e.g., bias_input)
            if key in key_to_index:
                return self.raw_groups[group_index][0][key_to_index[key]]
        elif name.endswith("_output"):
            key = name[:4]  # strip "_output" (e.g., bias_output)
            if key in key_to_index:
                if "sig" in name:
                    return getattr(self, f"{key}_output").replace(key, f"{key}sig")
                else:
                    return self.raw_groups[group_index][1][key_to_index[key]]
        raise AttributeError(f"Attribute {name} not found")
        
    def _parse_sci_list(self, group_index, dtype="input"):
        l = []
        for value in self.raw_groups[group_index][2].values():
            if dtype == "input":
                l += value[0]
            elif dtype == "output":
                l += value[1]
        return l

    def get_header(self, dtype):
        if dtype == "bias":
            header = fits.getheader(self.bias_input[0])
            header = prep_utils.write_IMCMB_to_header(header, self.bias_input)
            header["NFRAMES"] = len(self.bias_input)
        elif dtype == "dark":
            header = fits.getheader(self.dark_input[0])
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output] + self.dark_input)
            header["NFRAMES"] = len(self.dark_input)
        elif dtype == "flat":
            header = fits.getheader(self.flat_input[0])
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output, self.dark_output] + self.flat_input)
            header["NFRAMES"] = len(self.flat_input)
        return header

    def _calc_dark_scale(self):
        return 1

    def generate_masterframe(self):
        device_id = self.device_id

        self.logger.info(f"Generating masterframes for group {self._current_group+1} in GPU device {device_id}")
        st = time.time()
        
        for dtype in ["bias", "dark", "flat"]:
            
            self.logger.info(f"Generating {dtype} masterframe")
            input_data = getattr(self, f"{dtype}_input")
            header = self.get_header(dtype)

            if dtype == "bias":
                median, std = combine_images_with_cupy(input_data, device_id=device_id)
                self.bias_data = median

            elif dtype == "dark":
                median, std = combine_images_with_cupy(input_data, subtract=self.bias_data, device_id=device_id)
                self.dark_data = median
                n_sigma = self.config.preprocess.n_sigma
                self.generate_bpmask(median, n_sigma=n_sigma, header=header, device_id=device_id)

            elif dtype == "flat":
                dark_scale = self._calc_dark_scale()
                median, std = combine_images_with_cupy(input_data, 
                subtract=(self.bias_data + self.dark_data * dark_scale), norm=True, device_id=device_id)
                self.flat_data = median

            fits.writeto(
                getattr(self, f"{dtype}sig_output"),
                data=std,
                header=header,
                overwrite=True,
            )

            header = prep_utils.add_image_id(header)
            header = record_statistics(median, header, device_id=device_id)

            fits.writeto(
                getattr(self, f"{dtype}_output"),
                data=cp.asnumpy(median),
                header=header,
                overwrite=True,
            )
            
            del median
            cp.get_default_memory_pool().free_all_blocks()
        self.logger.info(f"Generation of masterframes completed in {time.time() - st:.2f} seconds")

    def data_reduction(self):
        st = time.time()

        n_head_blocks = self.config.preprocess.n_head_blocks
        use_multi_device = self.config.preprocess.use_multi_device
        
        if use_multi_device:
            num_devices = cp.cuda.runtime.getDeviceCount()
            device = np.arange(num_devices)
        else:
            num_devices = 1
            device = self.device_id
            
        batch_dist = calc_batch_dist(
            self.sci_input, num_devices=num_devices, use_multi_device=use_multi_device
        )

        self.logger.info(
            f"Processing {len(self.sci_input)} images in group {self._current_group+1} on GPU device(s): {device} "
        )

        threads = []
        results = [None] * num_devices
        start_idx = 0

        for batch in batch_dist:
            if sum(batch) == 0:
                break
            if use_multi_device:
                for device_id in range(num_devices):
                    end_idx = start_idx + batch[device_id]
                    self.logger.debug(f"Device {device_id} will process {end_idx - start_idx} images")
                    subset = self.sci_input[start_idx:end_idx]
                    t = threading.Thread(
                        target=process_batch_on_device,
                        args=(subset, self.bias_data, self.dark_data, self.flat_data, results, device_id),
                    )
                    t.start()
                    threads.append(t)
                    start_idx = end_idx
            else:
                end_idx = start_idx + batch[0]
                self.logger.debug(f"Device {device} will process {end_idx - start_idx} images")
                subset = self.sci_input[start_idx:end_idx]
                t = threading.Thread(
                    target=process_batch_on_device,
                    args=(subset, self.bias_data, self.dark_data, self.flat_data, results, device),
                )
                t.start()
                threads.append(t)
                start_idx = end_idx

            self.logger.debug("Data reduction is now processing on GPU device(s)")
            start_time = time.time()
            # Wait for all threads
            for t in threads:
                t.join()

            self.logger.debug(
                f"Data reduction has been completed in {time.time() - start_time:.0f} seconds for {len(self.sci_input)} iamges."
            )

        # Combine results
        all_results = [item for sublist in results if sublist for item in sublist]
        self.logger.info(
            f"Completed data reduction for {len(self.sci_input)} images in group {self._current_group+1} in {time.time() - st:.0f} seconds"
        )

        for idx, (raw_file, processed_file) in enumerate(zip(self.sci_input, self.sci_output)):
            header = fits.getheader(raw_file)
            header["SATURATE"] = prep_utils.get_saturation_level(header, self.bias_output, self.dark_output, self.flat_output)
            header = prep_utils.write_IMCMB_to_header(header, [self.bias_output, self.dark_output, self.flat_output, raw_file])
            add_padding(header, n_head_blocks, copy_header=False)
            fits.writeto(
                processed_file,
                data=all_results[idx],
                header=header,
                overwrite=True,
            )
        del all_results, self.bias_data, self.dark_data, self.flat_data
        self.logger.info(f"All images in group {self._current_group+1} are saved")
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    
    def generate_bpmask(self, data, n_sigma=5, header=None, device_id=0):
        mean, median, std = sigma_clipped_stats_cupy(data, sigma=3, maxiters=5, device_id=device_id)
        hot_mask = cp.abs(data - median) > n_sigma * std  # 1 for bad, 0 for okay
        hot_mask_cpu = cp.asnumpy(hot_mask).astype("uint8")
        del hot_mask, mean, median, std
        cp.get_default_memory_pool().free_all_blocks()
        
        newhdu = fits.CompImageHDU(data=hot_mask_cpu)
        if header:
            for key in ["INSTRUME", "GAIN", "EXPTIME", "EXPOSURE", "JD", "MJD", "DATE-OBS", "DATE-LOC", "XBINNING", "YBINNING"]:
                newhdu.header[key] = header[key]
            newhdu.header["COMMENT"] = "Header inherited from first dark frame"
        newhdu.header["NHOTPIX"] = (np.sum(hot_mask_cpu), "Number of hot pixels.")
        newhdu.header["SIGMAC"] = (n_sigma, "HP threshold in clipped sigma")
        newhdu.header["BADPIX"] = (1, "Pixel Value for Bad pixels")
        
        primary_hdu = fits.PrimaryHDU()
        newhdul = fits.HDUList([primary_hdu, newhdu])
        newhdul.writeto(self.bpmask_output, overwrite=True)

    def make_plots(self, group_index=None):
        st = time.time()
        if group_index is not None:
            group_index = self._current_group

        
        self.logger.info(f"Generating plots for master calibration frames of group {group_index+1}")
        use_multi_thread = self.config.preprocess.use_multi_thread
        
        plot_bias(self._get_raw_group("bias_output", group_index), savefig=True)
        mask = plot_bpmask(self._get_raw_group("bpmask_output", group_index), savefig=True)
        sample_header = fits.getheader(self._get_raw_group("bpmask_output", group_index), ext=1)
        if "BADPIX" in sample_header.keys():
            badpix = sample_header["BADPIX"]

        mask = mask != badpix
        fmask = mask.ravel()
        plot_dark(self._get_raw_group("dark_output", group_index), fmask, savefig=True)
        plot_flat(self._get_raw_group("flat_output", group_index), fmask, savefig=True)

        self.logger.info(f"Generating plots for science frames of group {group_index+1}")
        if use_multi_thread:
            threads = []
            for input_img, output_img in zip(self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index)):
                thread = threading.Thread(target=plot_sci, args=(input_img, output_img))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        else:
            for input_img, output_img in zip(self._get_raw_group("sci_input", group_index), self._get_raw_group("sci_output", group_index)):
                plot_sci(input_img, output_img)

        self.logger.info(f"Completed plot generation for {len(self._get_raw_group('sci_input', group_index))} images in group {group_index+1} in {time.time() - st:.2f} seconds")