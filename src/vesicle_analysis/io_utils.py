import os
from skimage.io import imsave
from aicsimageio import AICSImage

import numpy as np

import pandas as pd

__version__ = '0.1.0'


def get_channel(img: np.ndarray, ch: int):
    """
    Get individual channel from a mulitchannel image.

    Parameters:
    ----------
    img: multichannel numpy array
    ch: integer channel of interest (0-based)
    
    Returns:
    --------
    Single channel numpy array as type '<u2'

    """
    return img[:,:,:,ch].astype('<u2')

def read_image(path):
    '''
    open an tif or nd2

    Returns
    ------- 
    np.array image with axes TZYXC
    tuple with ZYX pixel size in um
    '''

    # only support nd2 and tif
    if not path.split('.')[-1] in ['nd2', 'tif', 'tiff']:
        raise NotImplementedError('.' + path.split('.')[-1] + ' images not supported.')

    # AICSImage.data is always TCZYX
    a = AICSImage(path)
    img = a.get_image_data("CZYX", T=0)
    # swap axes to ZYXC
    img = np.moveaxis(img, 0, -1)
    phy_size = a.physical_pixel_sizes
    assert phy_size.Y == phy_size.X, "X/Y pixel size is not the same!"
    zyx_resolution = (phy_size.Z, phy_size.Y, phy_size.X)

    return img, zyx_resolution


def save_data(path: str, tabel_to_save, nucleus_to_save, vesicle_to_save, more_info=''):
    """
    Custom save function

    Parameters
    -------
    path: string of input image
    tabel_to_save: pd.DataFrame
    nucleus_to_save: label image
    vesicle_to_save: label image
    more_info: string for additional filename information
    """
    filename = os.path.basename(path)
    folder = os.path.dirname(path)
    
    assert len(filename.split('.')) == 2, "You should not have dots ('.') in your file name. That's bad practise!"
    filename = filename.split('.')[0]

    # create new save path 
    new_path = os.path.join(folder, filename)

    # save and include script version in file name
    tabel_to_save.to_csv(new_path + '_' + more_info + '_v' + __version__ + '.csv')
    imsave(new_path + '_mask_nucleus_v' + __version__ + '.tif', nucleus_to_save)
    imsave(new_path + '_mask_vesicle_' + more_info + '_v' + __version__ + '.tif', vesicle_to_save)