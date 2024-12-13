import os

import numpy as np
from aicsimageio import AICSImage
from skimage.io import imsave


def get_pkg_version() -> str:
    """
    Get the vesicle-analysis package version.

    Return
    -------
    version number (str)

    """
    import vesicle_analysis as vsas

    return vsas.__version__


def get_channel(img: np.ndarray, ch: int):
    """
    Get individual channel from a multichannel image.

    Parameters
    ----------
    img: np.ndarray
        multichannel numpy array
    ch: integer
        integer channel of interest (0-based)

    Returns
    -------
    np.ndarray
        Single channel numpy array as type '<u2'
    """
    return img[:, :, :, ch].astype("<u2")


def read_image(path: str):
    """
    Open and tif or ND2.

    Returns
    -------
    np.array
        image with axes TZYXC
    tuple
        with ZYX pixel size in um
    """
    # only support ND2 and tif
    if path.split(".")[-1] not in ["nd2", "tif", "tiff"]:
        raise NotImplementedError("." + path.split(".")[-1] + " images not supported.")

    # AICSImage.data is always TCZYX
    a = AICSImage(path)
    img = a.get_image_data("CZYX", T=0)
    # swap axes to ZYXC
    img = np.moveaxis(img, 0, -1)
    phy_size = a.physical_pixel_sizes
    assert phy_size.Y == phy_size.X, "X/Y pixel size is not the same!"
    zyx_resolution = (phy_size.Z, phy_size.Y, phy_size.X)

    return img, zyx_resolution


def save_data(path: str, table_to_save, nucleus_to_save, vesicle_to_save, more_info=""):
    """
    Custom save function.

    Parameters
    ----------
    path: string
        path of input image
    table_to_save: pd.Dataframe
        table to be saved
    nucleus_to_save: np.ndarray
        label image to save
    vesicle_to_save: np.nd.array
        label image to save
    more_info: string
        for additional filename information

    Returns
    -------
    None
    """
    filename = os.path.basename(path)
    folder = os.path.dirname(path)

    assert len(filename.split(".")) == 2, """You should not have dots ('.') in your
    file name. That's bad practise!"""
    filename = filename.split(".")[0]

    # create new save path
    new_path = os.path.join(folder, filename)

    # save and include script version in file name
    table_to_save.to_csv(new_path + "_" + more_info + "_v" + get_pkg_version() + ".csv")
    imsave(new_path + "_mask_nucleus_v" + get_pkg_version() + ".tif", nucleus_to_save)
    imsave(
        new_path + "_mask_vesicle_" + more_info + "_v" + get_pkg_version() + ".tif",
        vesicle_to_save,
    )
