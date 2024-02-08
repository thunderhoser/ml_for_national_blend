"""Unit tests for nbm_utils.py"""

import unittest
import numpy
from ml_for_national_blend.utils import nbm_utils

TOLERANCE = 1e-6

ORIG_DATA_MATRIX_CHANNEL1 = numpy.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=float)

ORIG_DATA_MATRIX_CHANNEL2 = numpy.array([
    [9, 7, 5],
    [10, 10, 10],
    [-3, -2, 1]
], dtype=float)

ORIG_DATA_MATRIX = numpy.stack(
    [ORIG_DATA_MATRIX_CHANNEL1, ORIG_DATA_MATRIX_CHANNEL2], axis=-1
)

ORIG_X_COORD_MATRIX = numpy.array([
    [0.5, 2.5, 4.5],
    [0.5, 2.5, 4.5],
    [0.5, 2.5, 4.5]
])

ORIG_Y_COORD_MATRIX = numpy.array([
    [0.5, 0.5, 0.5],
    [1.5, 1.5, 1.5],
    [2.5, 2.5, 2.5]
])

NEW_X_COORDS = numpy.linspace(0, 6, num=7, dtype=float)
NEW_Y_COORDS = numpy.linspace(0, 3, num=4, dtype=float)

NAN = numpy.nan

INTERP_DATA_MATRIX_CHANNEL1 = numpy.array([
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [NAN, 2.75, 3.25, 3.75, 4.25, NAN, NAN],
    [NAN, 5.75, 6.25, 6.75, 7.25, NAN, NAN],
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN]
])

INTERP_DATA_MATRIX_CHANNEL2 = numpy.array([
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [NAN, 9, 8.5, 8.5, 8, NAN, NAN],
    [NAN, 3.75, 4, 4, 4.75, NAN, NAN],
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN]
])

INTERP_DATA_MATRIX_LINEAR = numpy.stack(
    [INTERP_DATA_MATRIX_CHANNEL1, INTERP_DATA_MATRIX_CHANNEL2], axis=-1
)

INTERP_DATA_MATRIX_CHANNEL1 = numpy.array([
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [NAN, 1, 2, 2, 3, NAN, NAN],
    [NAN, 4, 5, 5, 6, NAN, NAN],
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN]
])

INTERP_DATA_MATRIX_CHANNEL2 = numpy.array([
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN],
    [NAN, 9, 7, 7, 5, NAN, NAN],
    [NAN, 10, 10, 10, 10, NAN, NAN],
    [NAN, NAN, NAN, NAN, NAN, NAN, NAN]
])

INTERP_DATA_MATRIX_NN = numpy.stack(
    [INTERP_DATA_MATRIX_CHANNEL1, INTERP_DATA_MATRIX_CHANNEL2], axis=-1
)


class NbmUtilsTests(unittest.TestCase):
    """Each method is a unit test for nbm_utils.py."""

    def test_interp_data_to_nbm_grid_linear(self):
        """Ensures correct output from interp_data_to_nbm_grid.

        In this case, using linear interp.
        """

        this_interp_data_matrix = nbm_utils.interp_data_to_nbm_grid(
            data_matrix=ORIG_DATA_MATRIX,
            x_coord_matrix=ORIG_X_COORD_MATRIX,
            y_coord_matrix=ORIG_Y_COORD_MATRIX,
            use_nearest_neigh=False, test_mode=True,
            new_x_coords=NEW_X_COORDS, new_y_coords=NEW_Y_COORDS
        )

        self.assertTrue(numpy.allclose(
            this_interp_data_matrix, INTERP_DATA_MATRIX_LINEAR,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_interp_data_to_nbm_grid_nn(self):
        """Ensures correct output from interp_data_to_nbm_grid.

        In this case, using nearest-neighbour interp.
        """

        this_interp_data_matrix = nbm_utils.interp_data_to_nbm_grid(
            data_matrix=ORIG_DATA_MATRIX,
            x_coord_matrix=ORIG_X_COORD_MATRIX,
            y_coord_matrix=ORIG_Y_COORD_MATRIX,
            use_nearest_neigh=True, test_mode=True,
            new_x_coords=NEW_X_COORDS, new_y_coords=NEW_Y_COORDS
        )

        self.assertTrue(numpy.allclose(
            this_interp_data_matrix, INTERP_DATA_MATRIX_NN,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
