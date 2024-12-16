# Change log

version01: 20240522

 - optional background subtraction for vesicles

version02: 20240527

 - Change import to AICSImage (for pixel size metadata)
 - Adjust image file opening function accordingly
 - Nucleus fill holes
 - **Removed** binary erosion of the nucleus
 - Corrected object size filter on Nucleus

version03: 20240527

- refactor
- add batch possibility (at the end)

version04: 20240605

- add angle measurements (skipped if multiple nuclei in image)
- add nucleus centroid to vesicle centroid distance (skipped if multiple nuclei in image)
- batch loop saves TIFs of individual channels
- change threshold method from Otsu to Yen (FYI: background subtraction makes almost no difference)
- 20240607: fix vesicle size (convert pixels to um)
- 20240607: add variables for channels

version05: 20240610

- add csv file merging (after batch)
- 20240613: calculate minimal nucleus size using image calibration
- 20240619: correct table merging csv path for loading

version06: 20240717

- add convex hull area/volume measurement
- add circularity measurements, using regular perimeter and crofton perimeter
- add ellipse major and minor axes in calibrated units, and the ratio between minor/major axes
- add vesicle count per nucleus (i.e. total count of vesicles, multiple nuclei in same image will have same count)
- add the thickness of the analysed stack: (nZslices - 1) * z-step
- renamed column headers for "areas" to be volumes
- add 2D area of projected vesicle and nucleus
- 20240719: small vesicle size filter (smaller than 0.015um^3 = 5px with 0.12x0.16x0.16um pixel size)

version07: 20241126

- add additional string for saving with custom name (e.g. for adding the analysed vesicle channel)

version **0.1.0**: 20241216:

- changed from notebook to git repository (mix of python code files and short notebook)
- Changed filtering of nucleus image from median (ball=3) to gaussian with anisotropic sigma = 3
- add exception checks (e.g. when there is not enough vesicles to measure distances)
- checked on new images
  -   vesicles should be fine if I choose the lower threshold between Otsu and Yen
  -   except if total vesicle volume > 50% nucleus volume (then choose the other method)
  -   Nucleus segmentation is problematic, have to adjust min size (**from 75 to 25um**) and making new function with Gaussian and 'smarter' - TODO DESCRIBE!
- add angle normalisation when +/-180 -> -/+360


-------------

### Segmentation overview

Small crops of individual cells are expected for this workflow!

#### Nucleus segmentation

Nucleus channel is gaussian filtered with a sigma of 3 (anisotropy is calculated according to voxel size).
Otsu is used to threshold the filtered image.
Objects smaller than 25µm^3 are removed. In case, there are still more than 1, only the biggest object is kept.
(The workflow expects only one nucleus, to be able to calculate nucleus - vesicle distances).

#### Vesicle segmentation

Threshold on median filtered (ball=3) vesicle image with either Otsu or Yen method.
The method yielding the lower threshold value is chosen. Thresholding method can be forced/specified.
The methods are swapped, in case the total vesicle segmentation volume is bigger than 50% of the nucleus volume (even if method is specified).

The vesicles are refined with a watershed segmentation (using skimage.morphology.local_maxima), and
objects smaller than 0.015µm^3 are removed.