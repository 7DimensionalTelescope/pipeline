# gppy: Automated Pipeline for 7 Dimensional Telescope Image Processing

## Table of Contents
- [Overview](#overview)
- [Installation](#install)
- [Usage](#usage)
- [History and Development](#history-and-development)
- [Development Team](#development-team)
- [Version History](#version-history)
- [Contact](#contact)

## Overview
gppy is a comprehensive pipeline designed for processing optical astronomical images from 7 dimensional telescope (7DT), originally developed for IMSNG/GECKO ([GitHub](https://github.com/SilverRon/gppy)). The package implements a multi-threaded approach utilizing GPU and CPU multiprocessing to efficiently handle various stages of astronomical data processing, including:

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

## History and Development
Development of gpPy began in 2018 by Gregory Paek. The first stable release was published on September 1, 2023. The package underwent significant renovation in February 2025 to integrate with the 7DT pipeline system.

## Development Team
- **Original Developer**: Gregory Paek
- **Current Maintainers/Developers**: 
  - Donghwan Hyun
  - WonHyeong Lee
- **Core Contributors**:
  - Gregory Paek
  - Donggeun Tak
  
## Version History
- 2018: Initial development
- 2023.09.01: First public release
- 2025.02: Major renovation for 7DT pipeline integration

## Contact
If you have any inquiries or feedback, please contact the 7DT pipeline team via email at [7dt.pipeline@gmail.com](mailto:7dt.pipeline@gmail.com) or open an issue on our [GitHub repository](https://github.com/7DimensionalTelescope/pipeline).

