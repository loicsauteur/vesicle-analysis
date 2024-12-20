import numpy as np
import pandas as pd

import vesicle_analysis.io_utils as io
import vesicle_analysis.utils as uts


def test_something():
    pass


def test_io_version():
    # That's a stupid test. I don't remember why I put it here...
    assert (
        io.get_pkg_version() == "0.1.1"
    ), f"Version should be 0.1.0, but was {io.get_pkg_version()}"


def test_io_get_channel():
    mock = np.zeros((20, 20, 20, 5))
    out = io.get_channel(mock, 2)
    assert mock.shape[:-1] == out.shape


def test_get_angle_and_distance():
    # That is a very static test...
    nuc = {
        "label": [1],
        "centroid-0": [0],  # z is not relevant
        "centroid-1": [0],
        "centroid-2": [0],
    }
    ves = {
        "label": [1, 2, 3, 4],
        "centroid-0": [0, 0, 0, 0],  # z is not relevant
        "centroid-1": [10, 0, 8, 10],
        "centroid-2": [0, 10, 10, 2],
    }
    nuc = pd.DataFrame.from_dict(nuc)
    ves = pd.DataFrame.from_dict(ves)
    res = uts.get_angle_and_distance(nuc, ves, (1, 1, 1))
    # print(res['angle'])
    # print('middle', (max(res['angle']) + min(res['angle']))/2)
    # print('average', np.average(res['angle']))

    expected_angles = [90, 0, 38.66, 78.69]
    for angle, ref in zip(res["angle"], expected_angles):
        assert round(angle, 0) == round(ref, 0), (
            f"Angle {angle}° does not " f"match reference {ref}°."
        )
    middle_angle = (max(res["angle"]) + min(res["angle"])) / 2
    avg_angle = np.average(res["angle"])

    # assert the normalised angles (since between -180 and 180, no need to correct)
    expected_angles = [x - middle_angle for x in res["angle"]]
    assert expected_angles == res["angleNorm2Middle"], (
        f'Middle normalised angles are wrong. '
        f'{expected_angles} vs {res["angleNorm2Middle"]}'
    )
    expected_angles = [x - avg_angle for x in res["angle"]]
    assert expected_angles == res["angleNorm2Mean"], (
        f'Mean normalised angles are wrong. '
        f'{expected_angles} vs {res["angleNorm2Mean"]}'
    )
