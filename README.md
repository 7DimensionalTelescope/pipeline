# Py7DT.pipeline: Automated Pipeline for 7 Dimensional Telescope Image Processing

## Table of Contents
- [Overview](#overview)
- [Installation](#install)
- [Usage](#usage)
- [History and Development](#history-and-development)
- [Development Team](#development-team)
<!-- - [Version History](#version-history) -->
- [Contact](#contact)

## Overview
Py7DT.pipeline is a comprehensive pipeline designed for processing optical astronomical images from 7 dimensional telescope (7DT), originally developed for IMSNG/GECKO ([GitHub](https://github.com/SilverRon/gppy)). The package implements a multi-threaded approach utilizing GPU and CPU multiprocessing to efficiently handle various stages of astronomical data processing, including:

- Data reduction
- Astrometric calibration
- Photometric calibration
- Image stacking
- Image subtraction
- Automated transient detection

## Installation
We recommend using `conda` for installation:

```bash
conda env create -f environment.yml
```

This command will create a new environment named `pipeline` with all required packages listed in `environment.yml`. Activate this environment by running `conda activate pipeline`. Additionally, you will need to install external packages from [astromatic.net](https://www.astromatic.net/software/): `MissFITs`, `SCAMP`, `SWarp`, `SExtractor`, etc.

Then install the package itself and put the command-line scripts on your `PATH`:

```bash
pip install -e .                                          # so `import pipeline` works from any directory
echo 'export PATH="'"$(pwd)"'/pipeline/cli:$PATH"' >> ~/.bashrc
```

The editable install is what lets `pipeline/cli/*` import the package: Python puts the *script's*
directory on `sys.path`, not your working directory. The `PATH` line is for interactive use only —
the daemons build absolute paths from `SCRIPTS_DIR` and do not depend on it.

If the external binaries are installed under different names (`SWarp` rather than `swarp`, say), set
`SWARP_COMMAND` / `SEXTRACTOR_COMMAND` in `.env`; they take precedence over `ref/deployment.yml`.

### Host tuning
Post-install steps for a production host. Rationale and benchmarks are in `.claude/memory/performance-tuning.md`.

1. **Thread caps — nothing to do on a normal install.** `pipeline/__init__.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` to `1` and `NUMBA_NUM_THREADS` to `8` via `os.environ.setdefault()` before numpy/numba load, and `systemd/pipeline-queue.service` and `systemd/pipeline-trigger.service` carry the same values as `Environment=` lines (which also cover scripts that `import numpy` before `import pipeline`). `setdefault` means an explicit shell or unit variable still wins. On a **new host**, install the units and reload:
   ```bash
   ./systemd/fix_symlinks.sh   # symlinks units into /etc/systemd/system and runs daemon-reload
   sudo systemctl restart pipeline-trigger pipeline-queue
   ```
   Caveat: `fix_symlinks.sh` hardcodes `SYSTEMD_DIR="/home/pipeline-stable/pipeline/systemd"` (line 4), so it symlinks **that** path no matter which checkout you run it from — on a new host or a different clone, edit the variable or create the symlinks manually.
   Editing a unit file later requires `sudo systemctl daemon-reload` plus a restart; editing the Python default only reaches production once the `/home/pipeline-stable` checkout is updated.

2. **sysctl drop-in.** Raises the free-memory watermarks so nightly bulk processing is served by background reclaim instead of blocking in direct reclaim. `fix_symlinks.sh` does **not** install this file:
   ```bash
   sudo cp systemd/99-pipeline-tuning.conf /etc/sysctl.d/ && sudo sysctl --system
   sysctl vm.min_free_kbytes vm.watermark_scale_factor   # expect 4194304 and 200
   ```

3. **Swap priority.** Linux fills the highest-priority swap device first, so a small high-priority partition can saturate while a large swapfile sits unused. Check, and repoint if needed:
   ```bash
   swapon --show   # want the large swapfile at the highest PRIO
   sudo swapoff /dev/sda3 && sudo swapon -p 10 /data/swapfile
   ```
   `swapoff` faults everything on that device back into RAM, so run it **only when the queue is quiet**.

4. **Verify the install.**
   ```bash
   python -c "import pipeline, os, numba, threading; print({k: os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMBA_NUM_THREADS')}); print('numba:', numba.config.NUMBA_NUM_THREADS, 'threads:', threading.active_count())"
   ```
   Expect the five BLAS-family variables at `1`, `NUMBA_NUM_THREADS=8`, `numba: 8`, and `threads: 1`. A thread count in the dozens means an uncapped BLAS pool started first — check that the process really imported this `pipeline` package.

## History and Development
This pipeline is based on gpPy-gpu, whose predecessor, gpPy, was originally developed in 2018 by Gregory Paek. The first release of gpPy-gpu was on September 1, 2023.
gpPy-gpu underwent significant renovation to become a new package, Py7DT.pipeline, adding advanced orchestrating features. While the core part is named `pipeline`, Py7DT aims to be an encompassing framework of all 7DT-related tasks.


## Development Team
- **Current Maintainers/Developers**: 
  - Donghwan Hyun
  - Donggeun Tak
- **Core Contributors**:
  - Gregory Paek
  - Donghwan Hyun
  - Donggeun Tak
  - WonHyeong Lee
  
## Contact
If you have any inquiries or feedback, please contact the 7DT pipeline team via email at [7dt.pipeline@gmail.com](mailto:7dt.pipeline@gmail.com) or open an issue on our [GitHub repository](https://github.com/7DimensionalTelescope/pipeline).

