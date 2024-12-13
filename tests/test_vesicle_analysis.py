import numpy as np

import vesicle_analysis.io_utils as io


def test_something():
    pass


def test_io_version():
    assert (
        io.get_pkg_version() == "0.1.0"
    ), f"Version should be 0.1.0, but was {io.get_pkg_version()}"


def test_io_get_channel():
    mock = np.zeros((20, 20, 20, 5))
    out = io.get_channel(mock, 2)
    assert mock.shape[:-1] == out.shape
