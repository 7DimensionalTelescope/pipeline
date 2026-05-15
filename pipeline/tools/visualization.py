import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from ..preprocess.calc import bin_image


def quickvis(image_path: str = None, *, data: np.ndarray = None, binning: int = 4):
    if data is not None:
        _data_branch(data, binning)

    elif image_path is not None:
        if image_path.endswith(".fits"):
            data = fits.getdata(image_path)
            _data_branch(data, binning)
        else:
            _image_branch(image_path)

    else:
        raise ValueError("Either image_path or date must be provided")


def _data_branch(data, binning):
    binned_image = bin_image(data, binning)
    vmin, vmax = ZScaleInterval().get_limits(binned_image)
    plt.imshow(binned_image, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.show()


def _image_branch(image_path):
    from IPython.display import Image, display

    print(image_path)  # so that you can click it in VSCode
    display(Image(filename=image_path))
