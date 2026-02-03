import os
import re
import threading
import signal
import subprocess
import numpy as np
from astropy.io import fits
from typing import List, Tuple, Dict

from .errors import SolveFieldError, ScampError
from .services.logger import Logger

from .const import REF_DIR, SEXTRACTOR_COMMAND
from .utils import add_suffix, force_symlink, swap_ext, read_text_file, ansi_clean
from .utils.header import fitsrec_to_header


def sextractor(
    inim: str,
    outcat: str = None,
    sex_preset="prep",
    sex_options: Dict[str, str] = None,
    log_file: str = None,
    fits_ldac: bool = False,
    overwrite: bool = False,
    logger: Logger = None,
    return_sex_output=False,
    clean_log=True,
):
    """
    e.g., override default by supplying sex_args like {"-PIXEL_SCALE": f"{pixscale}"}.
    Sextractor log file is created in the same directory as outcat.
    No support for dual mode yet.
    """

    def get_sex_config(preset, ref_path=None):
        from .const import REF_DIR

        # "/data/pipeline_reform/gppy-gpu/gppy/ref/srcExt"
        ref_path = ref_path or os.path.join(REF_DIR, "srcExt")
        postfix = ["sex", "param", "conv", "nnw"]
        return [os.path.join(ref_path, f"{preset}.{pf}") for pf in postfix]

    # def chatter(message):
    #     if logger:
    #         logger.debug(message)
    #     else:
    #         print(message)
    def chatter(msg: str, level: str = "debug"):
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[sextractor:{level.upper()}] {msg}")

    # chatter("whoami", "info")
    # os.system("whoami")
    # os.system("echo $SHELL")
    # os.system("echo $TERM")

    default_outcat = (
        add_suffix(add_suffix(inim, sex_preset), "cat") if fits_ldac else swap_ext(add_suffix(inim, sex_preset), "cat")
    )
    outcat = outcat or default_outcat  # default is ascii.sextractor
    log_file = log_file or swap_ext(add_suffix(outcat, "sextractor"), "log")

    if os.path.exists(outcat) and not overwrite:
        chatter(f"Sextractor output catalog already exists: {outcat}, skipping...", "info")
        if return_sex_output:
            # take sexout from .log
            with open(log_file, "r") as f:
                lines = f.readlines()
                sexout = "\n".join(lines[1:])  # the first line is the command, not sexout
            return outcat, sexout
        return outcat

    sex_args_master = {}
    sex, param, conv, nnw = get_sex_config(sex_preset)
    sex_args_required = {
        "-c": f"{sex}",
        "-PARAMETERS_NAME": f"{param}",
        "-FILTER_NAME": f"{conv}",
        "-STARNNW_NAME": f"{nnw}",
        "-CATALOG_NAME": f"{outcat}",
    }

    sex_args_master.update(sex_args_required)

    if fits_ldac:
        sex_args_master["-catalog_type"] = "fits_ldac"
    if sex_options:
        sex_args_master.update(sex_options)  # override with user options

    options = [str(item) for k, v in sex_args_master.items() for item in (f"-{k}" if not k.startswith("-") else k, v)]
    sexcom = [SEXTRACTOR_COMMAND, f"{inim}"] + options

    chatter(f"Sextractor output catalog: {outcat}")
    chatter(f"Sextractor Log: {log_file}")

    sexcom = " ".join(sexcom)
    chatter(f"Sextractor Command: {sexcom}")

    if clean_log:
        sexcom = ansi_clean(sexcom)

    process = subprocess.Popen(
        sexcom,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        stdin=subprocess.DEVNULL,  # DEVNULL tricks SEx to think it's running non-interactively
    )
    stdout, _ = process.communicate()
    sexout = stdout if stdout else ""

    if process.returncode != 0:
        raise RuntimeError(f"Sextractor failed with return code {process.returncode}: {sexout}")

    with open(log_file, "w") as f:
        f.write(sexcom)
        f.write("\n" * 3)
        f.write(sexout)
        # f.write(sexerr)

    # Run the command and capture output
    # with open(log_file, "w") as f:
    #     process = subprocess.Popen(
    #         sexcom, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    #     )
    #     f.write(f"{sexcom}")

    #     output_lines = []
    #     for line in process.stdout:
    #         # print(line, end="")  # Print to console (optional)
    #         output_lines.append(line)
    #     process.wait()  # Ensure process completes
    #     sexout = "".join(output_lines)
    #     f.write(sexout)

    # This redirects all output to log_file
    # sexcom = f"{' '.join(sexcom)} > {log_file} 2>&1"
    # subprocess.run(sexcom, shell=True, text=True)

    # alternative; not working
    # with open(log_file, "w") as f:
    #     DEVNUL tricks SEx to think its running non-interactively
    #     subprocess.run(sexcom, stdout=f, stderr=f, text=True, stdin=subprocess.DEVNULL)

    chatter(f"Sextractor completed")

    if return_sex_output:
        return outcat, sexout
    else:
        return outcat


def solve_field(
    input_image: str = None,
    input_catalog: str = None,  # FITS_LDAC
    output_image: str = None,
    dump_dir: str = None,
    overwrite=False,
    solvefield_args: list = [],
    get_command=False,
    pixscale=0.505,
    radius=0.2,
    xcol="X_IMAGE",  # column name for X (pixels)
    ycol="Y_IMAGE",  # column name for Y (pixels)
    sortcol="MAG_AUTO",  # optional column to sort by
    sort_in_mag=True,  # sort in magnitude order
    timeout=60,  # subprocess timeout in seconds
    logger=None,
):
    """
    Runs Astrometry.net's `solve-field` to compute the World Coordinate System (WCS) for an input FITS image.

    This function creates a temporary working directory, generates a symbolic link to the input FITS file,
    and runs `solve-field` to solve the astrometry of the image. It supports real-time output streaming
    and optional command retrieval.

    Args:
        inim (str):
            Path to the input FITS image.
        outim (str):
            Path to the output FITS image.
        dump_dir (str, optional):
            Directory where intermediate results will be stored. If None, a temporary directory is created
            inside the input image's directory.
        get_command (bool, optional):
            If True, returns the command as a string instead of executing it. Defaults to False.
        pixscale (float, optional):
            Approximate pixel scale of the image in arcseconds per pixel. Defaults to 0.505 arcsec/pixel.
        radius (float, optional):
            Search radius for the solution in degrees. Defaults to 1.0 degree.
        xcol (str, optional):
            Column name for X (pixels). Defaults to "X_IMAGE". Needed for catalog input.
        ycol (str, optional):
            Column name for Y (pixels). Defaults to "Y_IMAGE". Needed for catalog input.
        sortcol (str, optional):
            Column name to sort by. Defaults to "MAG_AUTO".
        sort_in_mag (bool, optional):
            *Important*: the catalog must have the brightest object first. If the sorting key is in magnitude,
            setting this to True will add --sort-ascending. Defaults to True.

    Returns:
        str:
            If `get_command` is False, returns the path to the solved FITS file with the WCS solution.
            If `get_command` is True, returns the command string that would be executed.

    Raises:
        OSError: If the FITS file cannot be read.
        KeyError: If RA and DEC cannot be retrieved from the FITS header.

    Example:
        Solve an image normally:
        ```python
        solved_file = solve_field("image.fits")
        print(f"Solved FITS file: {solved_file}")
        ```
        If you have a sextractor catalog:
        ```python
        solved_file = solve_field(input_catalog="catalog.fits")
        print(f"Solved FITS file: {solved_file}")
        ```

        Get the command without executing it:
        ```python
        command = solve_field("image.fits", get_command=True)
        print(f"Command: {command}")
        ```
    """

    def chatter(msg: str, level: str = "debug"):
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[solve_field:{level.upper()}] {msg}")

    def set_input_output(input_image: str):
        nonlocal output_image
        nonlocal overwrite
        input_image = os.path.abspath(input_image)
        img_dir = os.path.dirname(input_image)
        working_dir = dump_dir or os.path.join(img_dir, "tmp_solvefield")
        working_dir = os.path.abspath(working_dir)
        if os.path.exists(working_dir) and overwrite:
            import shutil

            chatter(f"Removing working directory to overwrite: {working_dir}")
            shutil.rmtree(working_dir)
        os.makedirs(working_dir, exist_ok=True)

        # soft link inside working_dir
        fname = os.path.basename(input_image)
        soft_link = os.path.join(working_dir, fname)
        force_symlink(input_image, soft_link)

        # outname = os.path.join(working_dir, f"{Path(inim).stem}_solved.fits")
        output_image = output_image or os.path.join(os.path.splitext(soft_link)[0] + "_solved.fits")
        return soft_link, output_image

    def convert_ldac_to_xyls(infile, outfile=None, center=True, sortcol=None, all_columns=True):
        outfile = outfile or swap_ext(infile, ".xyls")

        with fits.open(infile) as hdul:
            data = hdul["LDAC_OBJECTS"].data  # or hdul[2]
            if center:
                image_header = fitsrec_to_header(hdul["LDAC_IMHEAD"].data)
        # data = Table.read(infile, format="ascii.sextractor")

        if all_columns:
            hdu = fits.BinTableHDU(data=data)
        else:
            cols = [
                fits.Column(name=xcol, format="E", array=data[xcol].astype(np.float32)),
                fits.Column(name=ycol, format="E", array=data[ycol].astype(np.float32)),
            ]
            if sortcol:
                cols.append(fits.Column(name=sortcol, format="E", array=data[sortcol].astype(np.float32)))

            hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(outfile, overwrite=True)
        if center:
            return outfile, image_header.get("RA", None), image_header.get("DEC", None)
        return outfile

    # Solve-field using the soft link
    if input_catalog:  # If user provided a Source Extractor catalog, use it instead of extracting sources
        soft_link, output_image = set_input_output(input_catalog)

        if os.path.exists(swap_ext(soft_link, ".solved")) and not overwrite:
            chatter(f"Solve-field already run: {swap_ext(soft_link, '.solved')}. Skipping...")
            return swap_ext(soft_link, ".wcs")

        soft_link, ra, dec = convert_ldac_to_xyls(soft_link, center=True, sortcol=sortcol)  # not a soft link anymore
        solvecom = [
            "solve-field", soft_link,  # "sources.xyls",
            "--x-column", xcol,
            "--y-column", ycol,
            "--width", "9576",  # we have LDAC_IMHEAD
            "--height", "6388",
            "--fields", "1",  # the extension to process. should point to IDAC_OBJECTS
            "--scale-unit", "arcsecperpix",
            "--scale-low", "0.49",
            "--scale-high", "0.52",
            "--crpix-center",
            "--no-plots",
        ]  # fmt: skip
        if ra and dec:
            solvecom += ["--ra", f"{ra:.4f}", "--dec", f"{dec:.4f}", "--radius", f"{radius:.1f}"]
        if sortcol:
            solvecom += ["--sort-column", sortcol]
            if sort_in_mag:
                solvecom += ["--sort-ascending"]
        # IMPORTANT: omit --use-source-extractor if we're providing a catalog

    elif input_image:
        soft_link, output_image = set_input_output(input_image)
        if os.path.exists(swap_ext(soft_link, ".solved")) and not overwrite:
            chatter(f"Solve-field already run: {swap_ext(soft_link, '.solved')}. Skipping...")
            return output_image

        # e.g., solve-field calib_7DT11_T00139_20250102_014643_m425_100s.fits --crpix-center --scale-unit arcsecperpix --scale-low '0.4949' --scale-high '0.5151' --no-plots --new-fits solved.fits --overwrite --use-source-extractor --cpulimit 30
        solvecom = [
            "solve-field", f"{soft_link}",  # this file is not changed by solve-field
            "--new-fits", output_image,  # you can give 'none'
            # "--config", f"{path_cfg}",
            # "--source-extractor-config", f"{path_sex_cfg}",
            # "--no-fits2fits",  # Do not create output FITS file
            "--overwrite",
            "--crpix-center",
            "--scale-unit", "arcsecperpix",
            "--scale-low", f"{pixscale*0.98}",
            "--scale-high", f"{pixscale*1.02}",
            "--use-source-extractor",  # Crucial speed boost. 30 s -> 5 s
            "--cpulimit", f"{30}",  # This is not about CORES. 
            "--no-plots",  # MASSIVE speed boost. 2 min -> 5 sec
            # "--no-tweak",  # Skip SIP distortion correction. 0.3 seconds boost.
            # "--downsample", "4",  # not much difference
        ]  # fmt: skip

        # give prior info on center
        try:
            ra = fits.getval(soft_link, "ra")
            dec = fits.getval(soft_link, "dec")
            solvecom = solvecom + [
                "--ra", f"{ra:.4f}",
                "--dec", f"{dec:.4f}",
                "--radius", f"{radius:.1f}",
            ]  # fmt: skip
        except:
            chatter("[WARNING] solve-field couldn't get RA Dec from header. Solving blindly")

    else:
        raise SolveFieldError.EmptyInputError("Either input_image or input_catalog must be provided")

    # custom arguments
    solvecom += solvefield_args

    if get_command:
        return " ".join(solvecom)

    # solvecom = f"{' '.join(solvecom)} > {log_file} 2>&1"
    # subprocess.run(solvecom, cwd=working_dir, shell=True)

    log_file = swap_ext(add_suffix(output_image, "solvefield"), "log")
    solvecom = " ".join(solvecom)
    # solveout = subprocess.getoutput(solvecom)
    process = subprocess.Popen(
        solvecom,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stdout and stderr
        text=True,
        bufsize=1,
    )

    # real-time output
    with open(log_file, "w") as f:
        f.write(solvecom + "\n" * 3)
        f.flush()

        try:
            for line in process.stdout:
                f.write(line)
                f.flush()
            # wait for process to finish within timeout
            process.wait(timeout=timeout)

        except subprocess.TimeoutExpired:
            chatter(f"Timeout reached ({timeout}s), terminating solve-field...")
            process.terminate()
            try:
                process.wait(5)  # give it a chance to terminate gracefully
            except subprocess.TimeoutExpired:
                chatter("solve-field did not terminate, killing...")
                process.kill()
            raise SolveFieldError.TimeoutError(f"solve-field timed out after {timeout}s.\n\tSee log: {log_file}")
    # except subprocess.CalledProcessError as e:
    #     chatter(f"solve-field failed: {e.returncode}")
    #     chatter(f"stderr output: {e.stderr.decode()}")
    #     raise e

    if not os.path.exists(swap_ext(soft_link, ".solved")):
        raise SolveFieldError.FileNotFoundError(
            f"Solve-field failed: {swap_ext(soft_link, '.solved')} not found.\n\tSee log: {log_file}"
        )

    if input_catalog:
        return swap_ext(soft_link, ".wcs")
    return output_image


def scamp(
    input: str,
    scampconfig=None,
    scamp_preset="prep",
    overwrite=True,
    ahead=None,
    path_ref_scamp=None,
    local_astref=None,
    get_command=False,
    clean_log=True,
    scamp_args: str = None,
    timeout=30,
    logger=None,
) -> List[str]:
    """
    Input is a fits-ldac catalog or a text file of those catalogs.
    Supply a text file of catalog filenames to run multiple catalogs jointly.

    scamp_preset: str = "prep" or "main"
    scampconfig: str = None, this overrides scamp_preset
    ahead: str = None
    path_ref_scamp: str = None
    local_astref: str = None
    get_command: bool = False
    clean_log: bool = True
    scamp_args: str = None
    timeout: int = 30  # based on the extremely dense field test of ~26 seconds
    """

    def chatter(msg: str, level: str = "debug"):
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[scamp:{level.upper()}] {msg}")

    scampconfig = scampconfig or os.path.join(REF_DIR, f"scamp_7dt_{scamp_preset}.config")
    # "/data/pipeline_reform/dhhyun_lab/scamptest/7dt.scamp"

    # "/data/pipeline_reform/dhhyun_lab/scamptest"
    log_file = os.path.splitext(input)[0] + "_scamp.log"

    # assumes joint run if input is not fits
    if os.path.splitext(input)[1] != ".fits":
        input_cat_list = read_text_file(input)
        input = f"@{input}"  # @ is astromatic syntax.
    else:
        input_cat_list = [input]

    output_list = [swap_ext(input_cat, "head") for input_cat in input_cat_list]
    if all([os.path.exists(head) for head in output_list]) and not overwrite:
        chatter(f"SCAMP output (.head) already exists: {output_list}\nSkipping...")
        return output_list

    # scampcom = f’scamp {catname} -c {os.path.join(path_cfg, “kmtnet.scamp”)} -ASTREF_CATALOG FILE -ASTREFCAT_NAME {gaialdac} -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {ahead} -STABILITY_TYPE INSTRUMENT’
    scampcom = f"scamp -c {scampconfig} {input}"

    # use the supplied astrefcat
    if local_astref and os.path.exists(local_astref):
        scampcom = f"{scampcom} -ASTREF_CATALOG FILE -ASTREFCAT_NAME {local_astref}"

    # download gaia edr3 refcat
    else:
        if not path_ref_scamp:
            path_ref_scamp = os.path.join(os.getcwd(), "ref_scamp")
            os.makedirs(path_ref_scamp, exist_ok=True)
        scampcom = f"{scampcom} -REFOUT_CATPATH {path_ref_scamp}"

    # supplied ahead file is merged to the input image header in fits_ldac hdu=1
    if ahead:
        # scampcom = f"{scampcom} -AHEADER_NAME {ahead}"
        scampcom = f"{scampcom} -AHEADER_GLOBAL {ahead}"

    if scamp_args:
        scampcom = f"{scampcom} {scamp_args}"

    # scampcom = f"{scampcom} >> {log_file} 2>&1"

    if get_command:
        return scampcom

    if clean_log:
        scampcom = ansi_clean(scampcom)

    # Save the command to the log file too
    with open(log_file, "w") as f:
        f.write(scampcom)
        f.write("\n" * 3)

    # old way
    # # scampcom = f"scamp -c {scampconfig} {outcat} -REFOUT_CATPATH {path_ref_scamp} -AHEADER_NAME {ahead_file}"
    # # subprocess.run(f"{scampcom} > {log_file} 2>&1", shell=True, text=True)
    # # astrefcat = f"{path_ref_scamp}/{obj}.fits" if 'path_astrefcat' not in upaths or upaths['path_astrefcat'] == '' else upaths['path_astrefcat']
    # # scamp_addcom = f"-ASTREF_CATALOG FILE -ASTREFCAT_NAME {astrefcat}"
    # # scamp_addcom = f"-REFOUT_CATPATH {path_ref_scamp}"
    # # try:
    # #     result = subprocess.run(scampcom, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # #     print(result.stdout.decode())
    # # except subprocess.CalledProcessError as e:
    # #     print(f"Command failed with error code {e.returncode}")
    # #     print(f"stderr output: {e.stderr.decode()}")

    num_stars_scamp_sees = None

    proc = subprocess.Popen(
        scampcom,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merged, preserves order
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,  # <-- new group so we can kill all children
    )

    # stream output concurrently so wait(timeout=...) can actually fire
    def pump():
        nonlocal num_stars_scamp_sees
        with open(log_file, "a", encoding="utf-8", errors="replace") as f:
            for line in iter(proc.stdout.readline, ""):
                f.write(line)
                f.flush()
                # parse "Group  1: 26/418 detections removed" to see how many stars scamp sees
                m = re.search(r"Group\s+\d+\s*:\s*\d+\s*/\s*(\d+)\s+detections removed", line)
                if m:
                    # number after "/" (26/418 -> 418)
                    num_stars_scamp_sees = int(m.group(1))
        proc.stdout.close()

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # kill the whole process group, not just the parent
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        finally:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        with open(log_file, "a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n[timed out after {timeout}s]\n")
        raise ScampError.TimeoutError(f"SCAMP timed out after {timeout}s. See log: {log_file}") from e

    finally:
        t.join(timeout=5)

    # =============== sanity check ===============

    # use parsed value from log
    if num_stars_scamp_sees is None:
        raise ScampError.ParseError(f"Could not find 'Group  1: X/Y detections removed' line in log: {log_file}")
    if num_stars_scamp_sees <= 5:  # more of a warning than an error, but useful in the pipeline context
        raise ScampError.NotEnoughSourcesError(
            f"Scamp sees {num_stars_scamp_sees} <= 5 detections to work with. See log: {log_file}"
        )

    # check if the output is valid
    for solved_head in output_list:
        if not os.path.exists(solved_head):
            raise ScampError.FileNotFoundError(
                f"SCAMP output (.head) does not exist: {solved_head}\nCheck Log file: {log_file}"
            )
        with open(solved_head, "r") as f:
            no_PV_terms = not any("PV1_0" in line for line in f)
        if no_PV_terms:
            raise ScampError.InvalidWcsSolutionError(f"SCAMP output ({solved_head}) is invalid. See log: {log_file}")

    return output_list


def missfits(inim):
    """
    Input images gets wcs updated, .back is made as a copy or the original
    Searches .head file in the same directory and with the same stem as inim and applies it to inim
    """

    missfitsconf = f"{REF_DIR}/7dt.missfits"
    missfitscom = f"missfits -c {missfitsconf} {inim}"
    # missfitscom = f"missfits -c {path_config}/7dt.missfits @{path_image_missfits_list}"

    process = subprocess.Popen(
        missfitscom,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process.wait()
    # working_dir = "/data/pipeline_reform/dhhyun_lab/scamptest/solvefield"
    # subprocess.run(missfitscom, shell=True, cwd=working_dir)


def swarp(
    input,
    output=None,
    center=None,
    overwrite=False,
    dump_dir=None,
    resample_dir=None,
    log_file=None,
    combine=True,
    weight_map=False,
    logger=None,
    swarp_args=None,
):
    """input is a list of filenames"""
    # if not input:
    #     raise ValueError("Input list is empty")
    # input = [os.path.abspath(f) for f in input]
    # working_dir = dump_dir or os.path.join(os.path.dirname(input[0]), "tmp_solvefield")

    from .utils import add_suffix

    def chatter(message):
        if logger:
            logger.debug(message)
        else:
            print(message)

    if input is list:
        input = ",".join(input)
    elif isinstance(input, str):  # assume file input
        input = f"@{input}"
    else:
        raise ValueError("Input must be a list or a string")

    if not center:
        raise ValueError("Deprojection center undefined")

    dump_dir = dump_dir or os.path.join(os.path.dirname(input), "tmp_swarp")
    log_file = log_file or os.path.join(dump_dir, "swarp.log")
    resample_dir = resample_dir or os.path.join(dump_dir, "resamp")
    os.makedirs(resample_dir, exist_ok=True)
    comim = output or os.path.join(dump_dir, "coadd.fits")
    # weightim = swap_ext(comim, "weight.fits")
    weightim = add_suffix(comim, "weight")

    if os.path.exists(comim) and not overwrite:
        if not (weight_map and not os.path.exists(weightim)):
            chatter(f"SWarp output image already exists: {comim}, skipping...")
            return

    # 	SWarp
    # swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_resamp} -GAIN_KEYWORD EGAIN -GAIN_DEFAULT {gain_default} -FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
    swarpcom = [
        "swarp", input,
        "-c", os.path.join(REF_DIR, '7dt.swarp'),
        "-IMAGEOUT_NAME", f"{comim}",
        "-WEIGHTOUT_NAME", f"{weightim}",
        "-CENTER_TYPE", "MANUAL",
        "-CENTER", f"{center} ",
        "-SUBTRACT_BACK", "N",
        "-RESAMPLE_DIR", f"{resample_dir}",
        # f"-GAIN_KEYWORD EGAIN -GAIN_DEFAULT {self.gain_default} "
        # f"-FSCALE_KEYWORD FAKE"
    ]  # fmt: skip

    if not weight_map:
        swarpcom.extend(["-WEIGHT_TYPE", "NONE"])

    if not combine:
        swarpcom.extend(["-COMBINE", "N"])

    swarpcom.extend(swarp_args or [])

    swarpcom = " ".join(swarpcom)
    chatter(f"SWarp Command: {swarpcom}")
    chatter(f"SWarp Log: {log_file}")

    process = subprocess.Popen(
        swarpcom,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    with open(log_file, "w") as f:
        f.write(swarpcom + "\n" * 3)
        f.flush()
        for line in process.stdout:
            f.write(line)
            f.flush()

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"SWarp failed with return code {process.returncode}. See log: {log_file}")

    return swarpcom


def hotpants(
    inim,
    tmplim,
    inmask,
    refmask,
    outim=None,
    out_conv_im=None,
    ssf=None,
    il=None,
    iu=None,
    tl=None,
    tu=None,
    nrx=None,
    nry=None,
    savexy=None,
    verbosity=0,
    log_file=None,
) -> str:

    n_sigma = 5
    header = fits.getheader(inim)

    # input Image
    il = il or header["SKYVAL"] - n_sigma * header["SKYSIG"]
    iu = iu or 60000

    # Template
    # tl, tu = refskyval - n_sigma * refskysig, 60000
    # tl, tu = refskyval - n_sigma * refskysig, 60000000
    # tl, tu = -20000, 5100000
    tl = tl or -60000000
    tu = tu or 60000000

    # x, y = 10200, 6800 for 7DT C3 images
    nrx = nrx or 3
    nry = nry or 2
    # nrx, nry = 1, 1
    # nrx, nry = 6, 4

    outim = outim or add_suffix(inim, "diff")
    out_conv_im = out_conv_im or add_suffix(inim, "conv")  # convolved sci image (oci)
    ssf = ssf or swap_ext(add_suffix(inim, "ssf"), ".txt")
    savexy = savexy or swap_ext(add_suffix(inim, "xy"), ".txt")

    hotpantscom = (
        f"hotpants -c t -n i "
        f"-iu {iu} -il {il} -tu {tu} -tl {tl} "
        f"-inim {inim} -tmplim {tmplim} -outim {outim} -oci {out_conv_im} "
        f"-imi {inmask} -tmi {refmask} "
        f"-savexy {savexy} "
        f"-v {verbosity} "
        f"-nrx {nrx} -nry {nry} "
        f"-ssf {ssf}"
    )
    log_file = log_file or os.path.join(os.path.dirname(out_conv_im), "hotpants.log")

    process = subprocess.Popen(
        hotpantscom,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout, _ = process.communicate()
    hotpantsout = stdout if stdout else ""

    if process.returncode != 0:
        raise RuntimeError(f"Hotpants failed with return code {process.returncode}: {hotpantsout}")

    with open(log_file, "w") as f:
        f.write(hotpantscom)
        f.write("\n" * 3)
        f.write(hotpantsout)
    # os.system(f"{hotpantscom} > {log_file} 2>&1")
    # print(hotpantscom)

    return hotpantscom
