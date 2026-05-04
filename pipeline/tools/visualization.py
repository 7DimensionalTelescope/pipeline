import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from ..preprocess.calc import bin_image


def quickvis(image_path: str = None, *, data: np.ndarray = None, binning: int = 4):
    if data is not None:
        pass
    elif image_path is not None:
        data = fits.getdata(image_path)
    else:
        raise ValueError("Either image_path or date must be provided")
    binned_image = bin_image(data, binning)
    vmin, vmax = ZScaleInterval().get_limits(binned_image)
    plt.imshow(binned_image, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.show()
