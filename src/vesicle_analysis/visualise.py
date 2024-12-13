import os
import napari.viewer
from skimage.io import imsave
from aicsimageio import AICSImage

import numpy as np
import napari
import math

# for segmentation and object/region analysis
import skimage.filters as filters
from skimage.morphology import disk, ball
import skimage.morphology as morphology
import skimage.segmentation as segmentation
import skimage.measure as measure

from skimage.measure import regionprops, regionprops_table
import pandas as pd

# for distance map (ndimage.distance_transform_edt)
from scipy import ndimage

__version__ = '0.1.0'

__viewer__ = None  # to refer to napari

def add_image(img: np.ndarray, name: str, colormap: str='gray'):
    """
    Add an image to the napari viewer

    Parameters
    -------
    img: numpy array, single channel
    name: string, for layer name
    colormap: string for LUT

    """
    global __viewer__
    if __viewer__ == None:
        #print('viewer is NONE')
        __viewer__ = napari.view_image(img, name=name, blending='additive', colormap=colormap)
    else:
        __viewer__.add_image(img, name=name, blending='additive', colormap=colormap)

def add_labels(img: np.ndarray, name: str):
    """
    Add a labels layer to the napari viewer

    Parameters
    -------
    img: numpy array
    name: string, for layer name

    """
    global __viewer__
    if __viewer__ == None:
        __viewer__ = napari.view_labels(img, name=name)
    else:
        __viewer__.add_labels(img, name=name)

def show_napari():
    global __viewer__
    if not __viewer__ == None:
        napari.run()
    else:
        raise(RuntimeError('Napari viewer not initialised'))
    