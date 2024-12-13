import napari
import napari.viewer
import numpy as np

# for segmentation and object/region analysis

# for distance map (ndimage.distance_transform_edt)

__viewer__ = None  # to refer to napari


def add_image(img: np.ndarray, name: str, colormap: str = "gray") -> None:
    """
    Add an image to the napari viewer.

    Parameters
    ----------
    img: numpy array
        single channel
    name: string
        for layer name
    colormap: string
        for LUT
    """
    global __viewer__
    __viewer__ = napari.viewer.current_viewer()

    if __viewer__ is None:
        # print('viewer is NONE')
        __viewer__ = napari.view_image(
            img, name=name, blending="additive", colormap=colormap
        )
    else:
        __viewer__.add_image(img, name=name, blending="additive", colormap=colormap)


def add_labels(img: np.ndarray, name: str) -> None:
    """
    Add a labels layer to the napari viewer.

    Parameters
    ----------
    img:
        numpy array
    name: string
        for layer name
    """
    global __viewer__
    __viewer__ = napari.viewer.current_viewer()

    if __viewer__ is None:
        __viewer__ = napari.view_labels(img, name=name)
    else:
        __viewer__.add_labels(img, name=name)


def show_napari() -> None:
    """
    Show napari.

    Currently, unused.
    """
    global __viewer__
    if __viewer__ is not None:
        napari.run()
    else:
        raise (RuntimeError("Napari viewer not initialised"))
