import math
import os

import numpy as np
import pandas as pd

# for distance map (ndimage.distance_transform_edt)
from scipy import ndimage
from skimage.measure import regionprops, regionprops_table

import vesicle_analysis.io_utils as io


def get_nucleus_volume(img_nucleus: np.ndarray, cal: tuple):
    """
    Get the volume of the segmented nucleus.

    Parameters
    ----------
    img_nucleus:
        label image of nucleus
    cal:
        tuple of ZYX pixel size in µm

    Returns
    -------
    Tuple:
        number of voxels, volume in calibrated units

    """
    props = regionprops(img_nucleus)
    if len(props) > 1:
        raise RuntimeError(
            f"Could not identify nucleus size, as there are {len(props)} objects"
        )
    elif len(props) == 0:
        raise RuntimeError("No objects in the image")
    return props[0].area, props[0].area * cal[0] * cal[1] * cal[2]


def min_max_distance(img_label: np.ndarray, calibration: tuple):
    """
    Finds the min and max distance between multiple object centroids (in µm).

    Parameters
    ----------
    img_label:
        label image of vesicles
    calibration:
        tuple of ZYX pixel size in µm

    Returns
    -------
    dictionary with:
        min/max distance per label
    """
    ves_ves_dist = {"label": [], "min_ves-ves-dist": [], "max_ves-ves-dist": []}
    ves_props = regionprops(img_label)
    if len(ves_props["label"]) == 0:
        # Should not come to this
        raise RuntimeError("Could not measure distances of <0> vesicles.")
    if len(ves_props["label"]) == 1:
        ves_ves_dist["label"].append(ves_props[0].label)
        ves_ves_dist["min_ves-ves-dist"].append("n/a")
        ves_ves_dist["max_ves-ves-dist"].append("n/a")
        return ves_ves_dist
    for cur_region in ves_props:
        # print(cur_region.label)
        cur_max_dist = 0
        cur_min_dist = 100000000000
        for region in regionprops(img_label):
            if not cur_region.label == region.label:
                x1 = cur_region.centroid[2] * calibration[1]
                y1 = cur_region.centroid[1] * calibration[1]
                z1 = cur_region.centroid[0] * calibration[0]
                x2 = region.centroid[2] * calibration[1]
                y2 = region.centroid[1] * calibration[1]
                z2 = region.centroid[0] * calibration[0]
                cur_dist = (
                    abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 + abs(z1 - z2) ** 2
                ) ** (1 / 2)
                if cur_dist > cur_max_dist:
                    cur_max_dist = cur_dist
                if cur_dist < cur_min_dist:
                    cur_min_dist = cur_dist
        ves_ves_dist["label"].append(cur_region.label)
        ves_ves_dist["min_ves-ves-dist"].append(cur_min_dist)
        ves_ves_dist["max_ves-ves-dist"].append(cur_max_dist)
    return ves_ves_dist


def distance_p2p(p1: tuple, p2: tuple, calibration: tuple):
    """
    With two 3D points, calculates the distance in calibrated units.

    Parameters
    ----------
    p1: tuple
        (Z, Y, X) coordinates
    p2: tuple
        (Z, Y, X) coordinates
    calibration: tuple
        of ZYX pixel size in µm

    Returns
    -------
    float:
        distance between p1 and p2 in µm
    """
    assert len(p1) == 3, "Point 1 has not 3 dimensions"
    assert len(p2) == 3, "Point 2 has not 3 dimensions"
    assert len(calibration) == 3, "Calibration is not for 3 dimensions"
    z1 = p1[0] * calibration[0]
    y1 = p1[1] * calibration[1]
    x1 = p1[2] * calibration[1]
    z2 = p2[0] * calibration[0]
    y2 = p2[1] * calibration[1]
    x2 = p2[2] * calibration[1]
    dist = (abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 + abs(z1 - z2) ** 2) ** (1 / 2)
    return dist


def get_angle_and_distance(
    df_nuc: pd.DataFrame, df_ves: pd.DataFrame, calibration: tuple
):
    """
    Calculates angles.

    With the centroid(s) of the nucleus and vesicles, the function will calculate the:
    - distance between nucleus and vesicle centroids in 3D (in µm)
    - calculate the angle (2D, i.e. projection) between nucleus and vesicle centroid
    - normalise the angle using the mean angle or the middle angle
      (middle between min and max anlges)
    - and absolute normalised angles (changing negative angles to positive)

    Parameters
    ----------
    df_nuc: pd.DataFrame
        containing the Nucleus centroids (1-3)
    df_ves: pd.DataFrame
        containing the vesicle centroids (1-3)
    calibration: tuple
        of ZYX pixel sizes in µm

    Returns
    -------
    dict: with values for vesicles,
          keys = [label, distance-NucCentroid, angle, angleNorm2Mean,
                  angleNorm2Middle, abs(angleNorm2Mean), abs(angleNorm2Middle)]
    None: if there is not exactly one nucleus
    """
    # Initialise result dictionary
    res = {
        "label": [],
        "distance-NucCentroid": [],
        "angle": [],
        "angleNorm2Mean": [],
        "angleNorm2Middle": [],
        "abs(angleNorm2Mean)": [],
        "abs(angleNorm2Middle)": [],
        # maybe later more to come
    }
    # do not calculate if there is more than one nucleus
    if len(df_nuc) != 1:
        print(
            ">> WARNING! there is not exactly one nucleus in the image ---",
            "skipping angle measurement <<",
        )
        return None

    # get the centroid of the nucleus
    z = df_nuc["centroid-0"].get(0)
    y = df_nuc["centroid-1"].get(0)
    x = df_nuc["centroid-2"].get(0)

    # if there is less than 2 vesicles, return empty dictionary
    if len(df_ves) < 2:
        return res

    # loop-over labels to get angle and distance
    for i in range(len(df_ves)):
        label = df_ves["label"].get(i)
        # get the centroid of the current vesicle
        z1 = df_ves["centroid-0"].get(i)
        y1 = df_ves["centroid-1"].get(i)
        x1 = df_ves["centroid-2"].get(i)
        # calculate the distance between the nucleus centroid and the vesicle one
        dis = distance_p2p((z, y, x), (z1, y1, x1), calibration=calibration)
        angle = math.degrees(math.atan2(y1 - y, x1 - x))
        res["label"].append(label)
        res["distance-NucCentroid"].append(dis)
        res["angle"].append(angle)

    # calculate angle mean etc.
    max_angle = max(res["angle"])
    min_angle = min(res["angle"])
    middle_angle = (min_angle + max_angle) / 2
    avg_angle = sum(res["angle"]) / len(res["angle"])

    # calculate per vesicle angle the deviation from (1) the average angle,
    # (2) the middle angle
    for i in range(len(df_ves)):
        cur_angle = res["angle"][i]
        res["angleNorm2Mean"].append(cur_angle - avg_angle)
        res["angleNorm2Middle"].append(cur_angle - middle_angle)
        res["abs(angleNorm2Mean)"].append(abs(cur_angle - avg_angle))
        res["abs(angleNorm2Middle)"].append(abs(cur_angle - middle_angle))
    return res


def measure_nucleus(nucleus_mask: np.ndarray, nucleus_channel: np.ndarray, cal: tuple):
    """
    Measure nucleus properties.

    Parameters
    ----------
    nucleus_mask:
        label image
    nucleus_channel:
        original raw intensity image
    cal:
        ZYX pixel size in µm

    Returns
    -------
    Fixme To Describe
    """
    if nucleus_mask is None:
        raise RuntimeError(
            "Nucleus mask is None. " "Cannot calculate nucleus properties."
        )
    voxel_size = cal[0] * cal[1] * cal[2]
    # Properties to measure
    props = (
        "label",
        "area",
        "area_filled",
        "area_convex",
        "centroid",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        # 'axis_major_length', 'axis_minor_length',
        # not for 3D: 'eccentricity', 'orientation',
        # 'perimeter', 'perimeter_crofton'
    )
    calibration_dict = {
        "label": 1,
        "area": voxel_size,
        "area_filled": voxel_size,
        "area_convex": voxel_size,
        "centroid-0": 1,
        "centroid-1": 1,
        "centroid-2": 1,
        "intensity_mean": 1,
        "intensity_max": 1,
        "intensity_min": 1,
        # 'axis_major_length':1, 'axis_minor_length':1,
        # not for 3D: 'eccentricity':1, 'orientation':1,
        # 'perimeter':1, 'perimeter_crofton':1
    }
    # Measure Nucleus props
    nuc_table = regionprops_table(
        nucleus_mask, intensity_image=nucleus_channel, properties=props
    )
    nuc_table = pd.DataFrame(nuc_table)
    nuc_table = nuc_table.mul(calibration_dict)

    # Calculate nucleus roundness in 2D (& ellipse axes)
    nuc_2d_props = regionprops(np.max(nucleus_mask, axis=0))
    nuc_circ_dict = {
        "label": [],
        "Total n vesicles": [],
        "Image stack height": [],
        "2D_circularity": [],
        "2D_circularity_crofton": [],
        "2D_ellipse_major": [],
        "2D_ellipse_minor": [],
        "ratio_2D_minor-major": [],
        "2D area": [],
    }
    for p in nuc_2d_props:
        # circularity if regular perimeter
        circ = 4 * math.pi * p.area / (p.perimeter**2)
        nuc_circ_dict["label"].append(p.label)
        nuc_circ_dict["2D_circularity"].append(circ)
        # circularity if crofton perimeter
        circ = 4 * math.pi * p.area / (p.perimeter_crofton**2)
        nuc_circ_dict["2D_circularity_crofton"].append(circ)
        # 2D elipse axes & ratio
        nuc_circ_dict["2D_ellipse_major"].append(p.axis_major_length * cal[1])
        nuc_circ_dict["2D_ellipse_minor"].append(p.axis_minor_length * cal[1])
        nuc_circ_dict["ratio_2D_minor-major"].append(
            p.axis_minor_length / p.axis_major_length
        )
        # add also the projected object area
        nuc_circ_dict["2D area"].append(p.area * cal[1] * cal[2])

        # TODO return merged or individual tables??


def measure_vesicles(ves_mask: np.ndarray, ves_channel: np.ndarray, cal: tuple):
    """
    Measure vesicle properties.

    Parameters
    ----------
    ves_mask:
        label image
    ves_channel:
        raw intensity image
    cal:
        ZYX pixel size

    Returns
    -------
    vesicle table: pd.DataFrame
        per label measurements of vesicles in 3D and 2D
    """
    if ves_mask is None:
        raise RuntimeError(
            "Vesicle mask is None. " "Cannot calculate vesicle properties."
        )
    voxel_size = cal[0] * cal[1] * cal[2]
    # Properties to measure
    props = (
        "label",
        "area",
        "area_filled",
        "area_convex",
        "centroid",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        # 'axis_major_length', 'axis_minor_length',
        # not for 3D: 'eccentricity', 'orientation',
        # 'perimeter', 'perimeter_crofton'
    )
    calibration_dict = {
        "label": 1,
        "area": voxel_size,
        "area_filled": voxel_size,
        "area_convex": voxel_size,
        "centroid-0": 1,
        "centroid-1": 1,
        "centroid-2": 1,
        "intensity_mean": 1,
        "intensity_max": 1,
        "intensity_min": 1,
        # 'axis_major_length':1, 'axis_minor_length':1,
        # not for 3D: 'eccentricity':1, 'orientation':1,
        # 'perimeter':1, 'perimeter_crofton':1
    }
    ves_table = regionprops_table(
        ves_mask, intensity_image=ves_channel, properties=props
    )
    ves_table = pd.DataFrame(ves_table)
    ves_table = ves_table.mul(calibration_dict)

    # 2D vesicle area measurements
    if len(ves_table["label"]) == 0:
        print(">>>>WARNING: No vesicles identified.<<<<")
    elif len(ves_table["label"]) > 100:
        print(
            ">>>>WARNING: More than 100 vesicles.",
            "kipped 2D vesicle measurements.<<<<",
        )
    else:
        ves_2D = {"label": [], "2D area": []}
        # project vesicles one by one
        for label in ves_table["label"]:
            single_label = np.max(np.where(ves_mask == label, ves_mask, 0), axis=0)
            single_props = regionprops(single_label)
            ves_2D["label"].append(label)
            ves_2D["2D area"].append(single_props[0].area * cal[1] * cal[2])
        # merge the 2D area measurements with the ves_table
        ves_table = pd.merge(ves_table, pd.DataFrame.from_dict(ves_2D), on="label")
    return ves_table


def measure_distances(nuc_mask: np.ndarray, ves_mask: np.ndarray, cal: tuple):
    """
    Measure distances and angles.

    Nucleus - vesicle distances
    Vesicle - vesicle distances
    Nucleus - vesicle angles

    Parameters
    ----------
    nuc_mask:
        label image
    ves_mask:
        label image
    cal:
        ZYX pixel size

    Returns
    -------
    FIXME to describe
    """
    if nuc_mask is None:
        raise RuntimeError("Cannot measure distances as nucleus mask is None.")
    if ves_mask is None:
        raise RuntimeError("Cannot measure distances as vesicle mask is None.")
    # Vesicle - Nucleus distance
    distance = ndimage.distance_transform_edt(nuc_mask < 1)  # FIXME not used yet
    dict_dist = {"label": [], "distance-NucSurface": []}  # FIXME not used yet

    # calculate vesicle to vesicle distance
    ves_ves_dist = min_max_distance(ves_mask, calibration=cal)

    # FIXME below just temporary
    return distance, dict_dist, ves_ves_dist


def do_measurements_old(
    nucleus_mask, nucleus_ch, vesicle_mask, vesicle_ch, calibration: tuple
):
    """
    Workflow for doing measurements.

    Measure results, includes:
    - also distance map creation for nucleus-vesicle distance
    - vesicle-vesicle centroid distance calculation
    - 2D nucleus and vesicle measurements (label projection),
      for area, circularity, ellipse minor/major
    - analysed stack height

    Parameters
    ----------
    nucleus_mask:
        label image
    nucleus_ch:
        intensity image
    vesicle_mask:
        label image
    vesicle_ch:
        intensity image
    calibration:
        tuple of ZYX pixel size in µm

    Returns
    -------
    pd.Dataframe:
        combined table for nucleus & vesicle measurements
    pd.Dataframe:
        table only of the nucleus measurements
    pd.Dataframe:
        table only of the vesicle measurements
    """
    # props to measure
    props = (
        "label",
        "area",
        "area_filled",
        "area_convex",
        "centroid",
        "intensity_mean",
        "intensity_max",
        "intensity_min",
        # 'axis_major_length', 'axis_minor_length',
        # not for 3D: 'eccentricity', 'orientation',
        # 'perimeter', 'perimeter_crofton'
    )
    voxel_size = calibration[0] * calibration[1] * calibration[1]
    calibration_dict = {
        "label": 1,
        "area": voxel_size,
        "area_filled": voxel_size,
        "area_convex": voxel_size,
        "centroid-0": 1,
        "centroid-1": 1,
        "centroid-2": 1,
        "intensity_mean": 1,
        "intensity_max": 1,
        "intensity_min": 1,
        # 'axis_major_length':1, 'axis_minor_length':1,
        # not for 3D: 'eccentricity':1, 'orientation':1, 'perimeter':1,
        # 'perimeter_crofton':1
    }

    # Nucleus measurements       ----------------------------------
    nuc_table = regionprops_table(
        nucleus_mask, intensity_image=nucleus_ch, properties=props
    )
    nuc_table = pd.DataFrame(nuc_table)

    # convert volumes to metric units
    # (not pixels -> basically only 'area' & 'area_filled' columns)
    nuc_table = nuc_table.mul(calibration_dict)

    # Calculate nucleus roundness in 2D, and ellipse axes
    nuc_2d_props = regionprops(np.max(nucleus_mask, axis=0))  # props of projection
    nuc_circ_dict = {
        "label": [],
        "Total n vesicles": [],
        "Image stack height": [],
        "2D_circularity": [],
        "2D_circularity_crofton": [],
        "2D_ellipse_major": [],
        "2D_ellipse_minor": [],
        "ratio_2D_minor-major": [],
        "2D area": [],
    }
    for p in nuc_2d_props:
        # circularity if regular perimeter
        circ = 4 * math.pi * p.area / (p.perimeter**2)
        nuc_circ_dict["label"].append(p.label)
        nuc_circ_dict["2D_circularity"].append(circ)
        # circularity if crofton perimeter
        circ = 4 * math.pi * p.area / (p.perimeter_crofton**2)
        nuc_circ_dict["2D_circularity_crofton"].append(circ)
        # 2D ellipse axes & ratio
        nuc_circ_dict["2D_ellipse_major"].append(p.axis_major_length * calibration[1])
        nuc_circ_dict["2D_ellipse_minor"].append(p.axis_minor_length * calibration[1])
        nuc_circ_dict["ratio_2D_minor-major"].append(
            p.axis_minor_length / p.axis_major_length
        )
        # add also the projected object area
        nuc_circ_dict["2D area"].append(p.area * calibration[1] * calibration[2])

    # Vesicle measurements        ----------------------------------
    ves_table = regionprops_table(
        vesicle_mask, intensity_image=vesicle_ch, properties=props
    )
    ves_table = pd.DataFrame(ves_table)

    # convert volumes to metric units (not pixels ->
    # basically only 'area' & 'area_filled' columns)
    ves_table = ves_table.mul(calibration_dict)

    # 2D vesicle area measurements                    >>>>>
    if len(ves_table["label"]) == 0:
        print(">>>>WARNING: No vesicles identified.<<<<<<")
    elif len(ves_table["label"]) <= 100:
        ves_2D = {"label": [], "2D area": []}
        # project vesicles one by one
        for label in ves_table["label"]:
            single_label = np.max(
                np.where(vesicle_mask == label, vesicle_mask, 0), axis=0
            )
            single_props = regionprops(single_label)
            ves_2D["label"].append(label)
            ves_2D["2D area"].append(
                single_props[0].area * calibration[1] * calibration[2]
            )
        # merge the 2D area measurements with the ves_table
        ves_table = pd.merge(ves_table, pd.DataFrame.from_dict(ves_2D), on="label")
    else:
        print(
            ">>>>WARNING: did not perform 2D vesicle measurement "
            "because there more than 100 vesicles in the image.<<<<<<"
        )

    # Vesicle - Nucleus distance                      >>>>>
    distance = ndimage.distance_transform_edt(nucleus_mask < 1)
    dict_dist = {"label": [], "distance-NucSurface": []}

    # calculate vesicle to vesicle distance           >>>>>
    ves_ves_dist = min_max_distance(vesicle_mask, calibration=calibration)

    # calculate the angles and merge the tables       >>>>>
    angle_dict = get_angle_and_distance(nuc_table, ves_table, calibration=calibration)

    for region in regionprops(vesicle_mask):
        dict_dist["label"].append(region.label)
        dict_dist["distance-NucSurface"].append(
            distance[int(region.centroid[0])][int(region.centroid[1])][
                int(region.centroid[2])
            ]
            * calibration[1]
        )

    # merge the tables                                >>>>>
    df_dist = pd.DataFrame.from_dict(dict_dist)
    ves_table = pd.merge(ves_table, df_dist, on="label")

    ves_table = pd.merge(ves_table, pd.DataFrame.from_dict(ves_ves_dist), on="label")

    if angle_dict is not None:
        ves_table = pd.merge(ves_table, pd.DataFrame.from_dict(angle_dict), on="label")

    # Finalise nucleus measurement table  --------------------------
    # Add the total vesicle count to the nucleus table (via nucleus circularity dict)
    nuc_circ_dict["Total n vesicles"] = [len(ves_table["label"])] * len(
        nuc_circ_dict["label"]
    )

    # add stack height in um to the nucleus table
    nuc_circ_dict["Image stack height"] = [
        (nucleus_ch.shape[0] - 1) * calibration[0]
    ] * len(nuc_circ_dict["label"])

    # Add the circularity to the nucleus table
    nuc_table = pd.merge(nuc_table, pd.DataFrame.from_dict(nuc_circ_dict), on="label")

    # concat nucleus and vesicle data ------------------------------

    df_image = pd.concat(
        [nuc_table, ves_table],
        keys=["Nuclei", "Vesicles"],
        names=["Object", "Object ID"],
    )
    # Rename the table headers for 'area' to volume
    df_image = df_image.rename(
        columns={
            "area": "volume",
            "area_filled": "volume (filled)",
            "area_convex": "volume (convex hull)",
        }
    )

    return df_image, nuc_table, ves_table


def merge_csv_files(dir_path: str, save_string: str = ""):
    """
    Merges several CSV files together.

    Looks for CSV files with the proper version tag in the name and
    merges them into a single table.

    Parameters
    ----------
    dir_path: String
        of path to folder
    save_string: (Optional) String
        for additional info in saving file name
    """
    all_dfs = []
    for file in os.listdir(dir_path):
        if file.endswith(io.get_pkg_version + ".csv") and not file.startswith(
            "Merged_tables_"
        ):
            loaded_df = pd.read_csv(os.path.join(dir_path, file))
            # add column with csv file name to the end of the DF
            loaded_df = loaded_df.assign(csv_file=file)
            all_dfs.append(loaded_df)
    merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    # Save merged table
    merged_name = "Merged_tables_" + save_string + "_v" + io.get_pkg_version + ".csv"
    merged.to_csv(os.path.join(dir_path, merged_name))
    print("Saved merged tables to:", os.path.join(dir_path, merged_name))
