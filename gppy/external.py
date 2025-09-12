import os
import subprocess
from astropy.io import fits
from .const import REF_DIR
from .utils import add_suffix, force_symlink, swap_ext, read_text_file, collapse, ansi_clean


def solve_field(
    input_image=None,
    input_catalog=None,
    output_image=None,
    dump_dir=None,
    get_command=False,
    pixscale=0.505,
    radius=1.0,
    sexcat=None,  # path to SExtractor catalog
    xcol="X_IMAGE",  # column name for X (pixels)
    ycol="Y_IMAGE",  # column name for Y (pixels)
    sortcol=None,  # optional column to sort by
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

        Get the command without executing it:
        ```python
        command = solve_field("image.fits", get_command=True)
        print(f"Command: {command}")
        ```
    """

    input_image = os.path.abspath(input_image)
    img_dir = os.path.dirname(input_image)
    working_dir = dump_dir or os.path.join(img_dir, "tmp_solvefield")
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # soft link inside working_dir
    fname = os.path.basename(input_image)
    soft_link = os.path.join(working_dir, fname)
    force_symlink(input_image, soft_link)

    # outname = os.path.join(working_dir, f"{Path(inim).stem}_solved.fits")
    output_image = output_image or os.path.join(os.path.splitext(soft_link)[0] + "_solved.fits")

    # Solve-field using the soft link
    # e.g., solve-field calib_7DT11_T00139_20250102_014643_m425_100s.fits --crpix-center --scale-unit arcsecperpix --scale-low '0.4949' --scale-high '0.5151' --no-plots --new-fits solved.fits --overwrite --use-source-extractor --cpulimit 30
    if input_catalog:
        solvecom = "solve-field sources.xyls   --fields 1   --x-column X_IMAGE --y-column Y_IMAGE   --width 9576 --height 6388   --scale-unit arcsecperpix --scale-low 0.49 --scale-high 0.52 --crpix-center"

    elif input_image:
        if sexcat is None:
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

        # If user provided a Source Extractor catalog, use it instead of extracting sources
        else:
            sexcat = os.path.abspath(sexcat)
            # Common SExtractor outputs work: FITS_LDAC (.cat), FITS table, or ASCII.
            # We point solve-field at it and specify which columns to read.
            solvecom += [
                "solve-field", sexcat,
                # "--x-column", xcol,
                # "--y-column", ycol,
            ]  # fmt: skip

            if sortcol:
                solvecom += ["--sort-column", sortcol]
            # IMPORTANT: omit --use-source-extractor if we're providing a catalog

    try:
        ra = fits.getval(input_image, "ra")
        dec = fits.getval(input_image, "dec")
        solvecom = solvecom + [
            "--ra", f"{ra:.4f}",
            "--dec", f"{dec:.4f}",
            "--radius", f"{radius:.1f}",
        ]  # fmt: skip
    except:
        print("Couldn't get RA Dec from header. Solving blindly")

    if get_command:
        return " ".join(solvecom)

    # # Use Popen for real-time output
    # process = subprocess.Popen(
    #     solvecom,
    #     cwd=working_dir,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     text=True,
    # )

    # # Also print messages to shell: should be captured by logger
    # for line in process.stdout:
    #     print(line, end="")
    # process.wait()  # Ensure the process completes

    # solvecom = f"{' '.join(solvecom)} > {log_file} 2>&1"
    # subprocess.run(solvecom, cwd=working_dir, shell=True)

    log_file = swap_ext(add_suffix(output_image, "solvefield"), "log")
    solvecom = " ".join(solvecom)
    solveout = subprocess.getoutput(solvecom)
    with open(log_file, "w") as f:
        f.write(solvecom)
        f.write("\n" * 3)
        f.write(solveout)
    if not os.path.exists(swap_ext(soft_link, ".solved")):
        raise FileNotFoundError(f"Solve-field failed: {swap_ext(soft_link, '.solved')} not found")
    return output_image


def scamp(
    input,
    scampconfig=None,
    scamp_preset="prep",
    ahead=None,
    path_ref_scamp=None,
    local_astref=None,
    get_command=False,
    clean_log=True,
    scamp_args: str = None,
) -> list[str]:
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
    """
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

    scampcom = f"{scampcom} >> {log_file} 2>&1"
    # print(scampcom)

    if get_command:
        return scampcom

    if clean_log:
        scampcom = ansi_clean(scampcom)

    # Save the command to the log file too
    with open(log_file, "w") as f:
        f.write(scampcom)
        f.write("\n" * 3)

    os.system(scampcom)
    # scampcom = f"scamp -c {scampconfig} {outcat} -REFOUT_CATPATH {path_ref_scamp} -AHEADER_NAME {ahead_file}"
    # subprocess.run(f"{scampcom} > {log_file} 2>&1", shell=True, text=True)

    # astrefcat = f"{path_ref_scamp}/{obj}.fits" if 'path_astrefcat' not in upaths or upaths['path_astrefcat'] == '' else upaths['path_astrefcat']
    # scamp_addcom = f"-ASTREF_CATALOG FILE -ASTREFCAT_NAME {astrefcat}"
    # scamp_addcom = f"-REFOUT_CATPATH {path_ref_scamp}"
    # try:
    #     result = subprocess.run(scampcom, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     print(result.stdout.decode())
    # except subprocess.CalledProcessError as e:
    #     print(f"Command failed with error code {e.returncode}")
    #     print(f"stderr output: {e.stderr.decode()}")
    solved_heads = []
    for input_cat in input_cat_list:
        solved_head = swap_ext(input_cat, "head")
        if not os.path.exists(solved_head):
            raise FileNotFoundError(f"SCAMP output (.head) does not exist: {solved_head}\nCheck Log file: {log_file}")
        solved_heads.append(solved_head)

    return solved_heads


def missfits(inim):
    """
    Input images gets wcs updated, .back is made as a copy or the original
    Searches .head file in the same directory and with the same stem as inim and applies it to inim
    """

    missfitsconf = f"{REF_DIR}/7dt.missfits"
    missfitscom = f"missfits -c {missfitsconf} {inim}"
    # missfitscom = f"missfits -c {path_config}/7dt.missfits @{path_image_missfits_list}"

    os.system(missfitscom)
    # working_dir = "/data/pipeline_reform/dhhyun_lab/scamptest/solvefield"
    # subprocess.run(missfitscom, shell=True, cwd=working_dir)


def sextractor(
    inim: str,
    outcat: str = None,
    se_preset="prep",
    log_file: str = None,
    fits_ldac: bool = False,
    sex_args: list = [],
    config=None,  # supply config.config
    logger=None,
    return_sex_output=False,
    clean_log=True,
):
    """
    e.g., override default by supplying sex_args like ["-PIXEL_SCALE", f"{pixscale}"]
    Sextractor log file is created in the same directory as outcat.
    No support for dual mode yet.
    """

    def get_sex_config(preset, ref_path=None):
        from .const import REF_DIR

        # "/data/pipeline_reform/gppy-gpu/gppy/ref/srcExt"
        ref_path = ref_path or os.path.join(REF_DIR, "srcExt")
        postfix = ["sex", "param", "conv", "nnw"]
        return [os.path.join(ref_path, f"{preset}.{pf}") for pf in postfix]

    def chatter(message):
        if logger:
            logger.debug(message)
        else:
            print(message)

    if config:
        chatter("Using Configuration Class")
        sex = config.sex.sex
        param = config.sex.param
        nnw = config.sex.nnw
        conv = config.sex.conv
    else:
        sex, param, conv, nnw = get_sex_config(se_preset)

    default_outcat = (
        add_suffix(add_suffix(inim, se_preset), "cat") if fits_ldac else swap_ext(add_suffix(inim, se_preset), "cat")
    )
    outcat = outcat or default_outcat  # default is ascii.sextractor
    log_file = log_file or swap_ext(add_suffix(outcat, "sextractor"), "log")

    sexcom = [
        "source-extractor", f"{inim}",
        "-c", f"{sex}",
        "-CATALOG_NAME", f"{outcat}",
        # "-catalog_type", "fits_ldac",  # this is for scamp presex
        "-PARAMETERS_NAME", f"{param}",
        "-FILTER_NAME", f"{conv}",
        "-STARNNW_NAME", f"{nnw}",
    ]  # fmt: skip

    if fits_ldac:
        sexcom.extend(["-catalog_type", "fits_ldac"])

    # add additional arguments when given
    sexcom.extend(sex_args or [])

    chatter(f"Sextractor output catalog: {outcat}")
    chatter(f"Sextractor Log: {log_file}")

    sexcom = " ".join(sexcom)
    chatter(f"Sextractor Command: {sexcom}")

    if clean_log:
        sexcom = ansi_clean(sexcom)

    sexout = subprocess.getoutput(sexcom)
    # result = subprocess.run(sexcom, shell=True, capture_output=True, text=True, stderr=subprocess.STDOUT)
    # sexout = result.stdout + result.stderr
    with open(log_file, "w") as f:
        f.write(sexcom)
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


def swarp(
    input,
    output=None,
    center=None,
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
    os.system(f"{swarpcom} > {log_file} 2>&1")
    # os.system(swarpcom)

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
    log_file=None,
):
    """
    il, iu: input image lower/upper limits
    tl, tu: template image lower/upper limits
    nrx, nry: number of image regions in x/y dimension.
    ssf: substamp file
    """

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

    hotpantscom = (
        f"hotpants -c t -n i "
        f"-iu {iu} -il {il} -tu {tu} -tl {tl} "
        f"-inim {inim} -tmplim {tmplim} -outim {outim} -oci {out_conv_im} "
        f"-imi {inmask} -tmi {refmask} "
        f"-v 0 "
        f"-nrx {nrx} -nry {nry} "
        f"-ssf {ssf}"
    )
    log_file = log_file or os.path.join(os.path.dirname(out_conv_im), "hotpants.log")

    hotpantsout = subprocess.getoutput(hotpantscom)
    with open(log_file, "w") as f:
        f.write(hotpantscom)
        f.write("\n" * 3)
        f.write(hotpantsout)
    # os.system(f"{hotpantscom} > {log_file} 2>&1")
    # print(hotpantscom)

    return hotpantscom
