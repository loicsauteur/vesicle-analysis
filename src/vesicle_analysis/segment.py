import numpy as np

import skimage.filters as filters
from skimage.morphology import disk, ball
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.measure as measure
from skimage.measure import regionprops
# for distance map (ndimage.distance_transform_edt)
from scipy import ndimage

from typing import Optional

__version__ = '0.1.0'


def segment_nucleus(dapi_channel: np.ndarray, calibration: tuple):
    """
    Segment nucleus:
    - Gaussian filter with sigma of 3
    - Otsu threshold
    - removing objects smaller than 25µm^3 (if multiple)
    - if after removing small objects there are still more than 1, just keep the biggest one

    Parameters:
    ----------
    dapi_channel: numpy array of single channel (dapi)
    calibration = tuple of ZYX pixle size in µm

    Returns
    -------
    Tuple of:
        label image: numpy array | None
        num detections: Integer
    
    """
    # 3D gaussian with sigma 3 (scaled for axes)
    sigma = 3
    z_scaling = calibration[0] / calibration[1]
    nuc = filters.gaussian(dapi_channel, (sigma * z_scaling, sigma, sigma))

    # median fitler on the nucleus channel
    #nuc_median = filters.rank.median(dapi_channel, footprint=ball(3))
    
    # otsu threshold of nucleus
    nuc = nuc >= filters.threshold_otsu(nuc)
    # fill holes
    nuc = ndimage.morphology.binary_fill_holes(nuc)
    # label the image
    nuc, num = morphology.label(nuc, connectivity=1, return_num=True)
    
    if num == 0:
        return (None, num)
    elif num == 1:
        return (nuc, num)
    else:
        # If more than 1 object => exclude small objects (min_size = 25um^3)
        minimal_size = 25 / (calibration[0] * calibration[1] * calibration[2])
        nuc = morphology.remove_small_objects(nuc, min_size=minimal_size)
        _, num = morphology.label(nuc, connectivity=1, return_num=True)
        if num > 1:
            # Select only the biggest region
            print("More than 1 nucleus object after size filter. Will keep only the biggest object.")
            props = regionprops(nuc)
            size = 0
            label = 0
            for p in props:
                if p.area >= size:
                    size = p.area
                    label = p.label
            nuc = np.where(nuc == label, 1, 0)
        return (nuc, num)
    

def segment_vesicles(vesicle_channel: np.ndarray, cal: tuple, nuc_voxels: int, method: Optional[str], troubleshoot: bool=False):
    """
    Segment vesicles:
    On median filtered (ball=3) image, uses either Otsu or Yen threshold (whichever is lower), if not specified.
    Then uses a seeded watershed.
    Excludes objects smaller than 0.015µm^3.
    In case of oversegmentation, i.e. total vesicle volume > 50% of nucleus, segmentation method will be swapped.

    Parameters
    -------
    vesicle_channel: single channel np.array
    cal = tuple of ZYX pixle size in µm 
    nuc_voxels: Integer number of voxels of the nucleus
    method = String for thresholding method, default = None. Implemented only otsu and yen.
             If None, lower threshold (otsu or yen) is chosen.
    troubleshoot: (bool) will return all the segemntation steps
    
    Returns
    -------
    Tuple:
    Label image: np.array or None (if no vesicles),
    Number of objects: Integer
    Thresholding method: String

    IF troubleshoot = True, returns Tuple of:
    ves_median (gray-scale np.array), maxima (label np.array), 
    binary (np.array), watershed (label np.array), thresholding method (String)

    """
    # Check the method if inputed
    if not method == None and method.lower() not in ['otsu', 'yen']:
        raise NotImplementedError(f'Thresholding vesicles with {method} is not implemented, please use "otsu" or "yen".')

    # Median filter of vesile channel with ball size 3
    ves_median = filters.rank.median(vesicle_channel, footprint=ball(3))

    # Auto-choose method for segmentation
    if method == None:
        if filters.threshold_otsu(ves_median) <= filters.threshold_yen(ves_median):
            method = 'otsu'
        else:
            method = 'yen'
    else:
        # Override auto-chooser with parameter method
        method = method.lower()
    
    if method == 'otsu':
        threshold = filters.threshold_otsu(ves_median)
    else:
        threshold = filters.threshold_yen(ves_median)

    # Do the watershed
    maxima = morphology.local_maxima(ves_median, allow_borders=False)
    maxima = measure.label(maxima)
    binary = ves_median >= threshold  # binarization
    maxima = np.where(binary > 0, maxima, 0)  # keep maxima only where objects
    watershed = segmentation.watershed(binary, markers=maxima, mask=binary)

    # Exclude small objects (min_size = 0.015µm^3 = <5px with 0.12x0.16x0.16 voxel size)
    minimal_size = 0.015 / (cal[0] * cal[1] * cal[2])
    watershed = morphology.remove_small_objects(watershed, min_size=minimal_size)

    # Check that the total volume of vesicles is less than half the nucleus, i.e. not oversegmented
    volume = 0
    for p in props:
        volume += p.area
    if volume > 0.5 * nuc_voxels:
        # Swap thresholding methods
        if method == 'yen':
            method = 'otsu'
            threshold = filters.threshold_otsu(ves_median)
        else:
            method = 'yen'
            threshold = filters.threshold_yen(ves_median)
        # Redo watershed
        binary = ves_median >= threshold
        maxima = morphology.local_maxima(ves_median, allow_borders=False)
        maxima = measure.label(maxima)
        maxima = np.where(binary > 0, maxima, 0)
        watershed = segmentation.watershed(binary, markers=maxima, mask=binary)
        watershed = morphology.remove_small_objects(watershed, min_size=minimal_size)

    # For troubleshooting -> return all intermediate images...
    if troubleshoot:
        return ves_median, maxima, binary, watershed, method

    # Check the number of vesicles
    props = measure.regionprops(watershed)
    if len(props) == 0:
        return None, 0, method
    elif len(props) == 1:
        return watershed, 1, method
    else:
        return watershed, len(props), method



def segment_vesicles_old(vesicle_channel, calibration: tuple, method: str='yen', do_bg_corr: bool=False):
    """
    Sement vesicles
    - optional background correction (tophat, with ball=5px (empiric))
    - median filter with ball of radius 3
    - yen threshold
    - seeded watershed

    Parameters
    -------
    vesicle_channel: single channel np.array
    calibration = tuple of ZYX pixle size in µm 
    method = String for thresholding method, default = 'yen'. Implemented only otsu and yen.
    do_bg_corr: (bool) whether to do a 'rolling ball' background subtraction
    
    Returns
    -------
    label image: np.array
    """
    # Check if selected method is implemented
    if method.lower() not in ['otsu', 'yen']:
        raise NotImplementedError(f'Thesholding vesicles with {method} is not implemented, please use "otsu" or "yen".')
    
    # Background subtraction for vesicle marker
    if do_bg_corr:
        ves_ball = ball(5)
        img_ves_copy = morphology.white_tophat(vesicle_channel, footprint=ves_ball)
    else:
        img_ves_copy = vesicle_channel
    
    # filter vesicle channel with median filter
    ves_median = filters.rank.median(img_ves_copy, footprint=ball(3))
    # detect local maxima for watershed
    maxima = morphology.local_maxima(ves_median, allow_borders=False)
    maxima = measure.label(maxima)
    # threshold median filtred vesicles with yen
    if method.lower() == 'otsu':
        ves_median = ves_median >= filters.threshold_otsu(ves_median)
    elif method.lower() == 'yen':
        ves_median = ves_median >= filters.threshold_yen(ves_median)
    # keep only the maxima where also vesicle objects are
    maxima = np.where(ves_median > 0, maxima, 0)
    # watershed the vesicles
    ves_median = segmentation.watershed(ves_median, markers=maxima, mask=ves_median)
    # exclude small objects (min_size = 0.015um^3 = which is <5px with 0.12x0.16x0.16um pixel size)
    minimal_size = 0.015 / (calibration[0] * calibration[1] * calibration[2])
    ves_median = morphology.remove_small_objects(ves_median, min_size=minimal_size)
    return ves_median

def segment_nucleus_old(dapi_channel, calibration: tuple):
    """
    Segment nucleus
    - median filter with ball of radius 3
    - otsu threshold
    - remvoing small objects (<25µm^2)

    Parameters
    -------
    dapi_channel: single channel np.array
    calibration = tuple of ZYX pixle size in µm
    
    Returns
    -------
    label image: np.array
    """
    # median fitler on the nucleus channel
    nuc_median = filters.rank.median(dapi_channel, footprint=ball(3))
    # otsu threshold of nucleus
    nuc_median = nuc_median >= filters.threshold_otsu(nuc_median)
    # fill holes
    nuc_median = ndimage.morphology.binary_fill_holes(nuc_median)
    # binary erosion of nucleus object ---DISABLED---
    #nuc_median = morphology.binary_erosion(nuc_median)
    # label the image
    nuc_median = morphology.label(nuc_median, connectivity=1)
    # exclude small objects (min_size = 75um^3)
    minimal_size = 75 / (calibration[0] * calibration[1] * calibration[2])
    nuc_median = morphology.remove_small_objects(nuc_median, min_size=minimal_size)
    return nuc_median
