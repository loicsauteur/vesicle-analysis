import os

import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from skimage.io import imsave


def get_pkg_version() -> str:
    """
    Get the vesicle-analysis package version.

    Return
    -------
    version number (str)

    """
    import vesicle_analysis as vs_as

    return vs_as.__version__


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


def combine_csv_in_folder(path: str, more_info: str):
    """
    Combine all CSVs in a folder.

    Will only combine the files with the current package version.

    Parameters
    ----------
    path
        String path to folder. If a file, will take the parent.
    more_info:
        String for additional filename information (on vesicle)

    Returns
    -------
    None
    """
    if os.path.isfile(path):
        print(f"Path was a file, will take the parent of: {path}")
        path = os.path.dirname(path)
    # Load the csvs as DataFrames
    all_dfs = []
    for file in os.listdir(path):
        if file.endswith("v" + get_pkg_version() + ".csv") and not file.startswith(
            "Merged_tables_"
        ):
            df = pd.read_csv(os.path.join(path, file))
            df = df.assign(csv_file=file)
            all_dfs.append(df)
    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    merged_path = "Merged_tables_" + more_info + "_v" + get_pkg_version() + ".csv"
    merged_path = os.path.join(path, merged_path)
    merged.to_csv(merged_path)
    print("Merged all csv files to:", merged_path)


def save_data(
    path_file: str,
    table: pd.DataFrame,
    nuc_mask: np.ndarray,
    ves_mask: np.ndarray,
    nuc_ch: np.ndarray,
    ves_ch: np.ndarray,
    more_info: str,
    save_raw_channels: bool = True,
):
    """
    Saves all results.

    Table, channels and masks. 'more_info' is for describing the vesicle channel,
    e.g. what marker.

    Parameters
    ----------
    path_file
        String path to the raw image, parent will be used.
    table
        DataFrame to save
    nuc_mask
        Nucleus mask image to save
    ves_mask
        Vesicle mask image to save
    nuc_ch
         Nucleus single channel image to save
    ves_ch
        Vesicle single channel image to save
    more_info
        String for additional filename information (on vesicle)
    save_raw_channels
        Whether to save the raw channel images. Default = True.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError: if there are dots in the filename
    """
    filename = os.path.basename(path_file)
    folder = os.path.dirname(path_file)

    # Create saving path
    filename = filename.split(".")
    if len(filename) != 2:
        raise RuntimeError('You should not have dots (".") in your file name.')
    filename = filename[0]
    path = os.path.join(folder, filename)

    # Save elements
    table.to_csv(path + "_" + more_info + "_v" + get_pkg_version() + ".csv")
    imsave(
        path + "_mask_nucleus_v" + get_pkg_version() + ".tif",
        nuc_mask,
        check_contrast=False,
    )
    imsave(
        path + "_mask_vesicle_" + more_info + "_v" + get_pkg_version() + ".tif",
        ves_mask,
        check_contrast=False,
    )
    if save_raw_channels:
        imsave(path + "_nucCh.tif", nuc_ch, check_contrast=False)
        imsave(path + "_vesCh.tif", ves_ch, check_contrast=False)
    print(f"   Saved data to: {path}*")


def save_data_old(
    path: str, table_to_save, nucleus_to_save, vesicle_to_save, more_info=""
):
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
