"""Unit tests for wrf_arw_utils.py"""

import unittest
import numpy
import xarray
from ml_for_national_blend.utils import wrf_arw_utils

TOLERANCE = 1e-6

# The following constants are used to test precip_from_incremental_to_full_run.
INCREMENTAL_PRECIP_VALUES = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    5, 6, 7, 8, 9
], dtype=float)

ACCUMULATED_PRECIP_VALUES = numpy.array([
    0,
    1, 3, 4, 6, 7, 9, 10, 12, 13, 15,
    16, 18, 19, 21, 22, 24, 25, 27, 28, 30,
    31, 33, 34, 36, 37, 39, 40, 42, 43, 45,
    48, 52, 55, 59, 62, 66, 69, 73, 76, 80, 83, 87,
    92, 98, 105, 113, 122
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    wrf_arw_utils.FORECAST_HOUR_DIM: wrf_arw_utils.ALL_FORECAST_HOURS,
    wrf_arw_utils.ROW_DIM: numpy.array([0], dtype=int),
    wrf_arw_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    wrf_arw_utils.FIELD_DIM: [wrf_arw_utils.PRECIP_NAME]
}

THESE_DIM = (
    wrf_arw_utils.FORECAST_HOUR_DIM, wrf_arw_utils.ROW_DIM,
    wrf_arw_utils.COLUMN_DIM, wrf_arw_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    wrf_arw_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (wrf_arw_utils.ROW_DIM, wrf_arw_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    wrf_arw_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    wrf_arw_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

WRF_ARW_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

# The following constants are used to test remove_negative_precip.
ACCUMULATED_PRECIP_VALUES_WITH_NEG = numpy.array([
    numpy.nan,
    1, 2, 3, 4, 5, 4, numpy.nan, 8, 9, 10,
    11, 12, 13, 14, 15, 13, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 22, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 31, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 40, 47
], dtype=float)

ACCUMULATED_PRECIP_VALUES_SANS_NEG = numpy.array([
    numpy.nan,
    1, 2, 3, 4, 5, 5, 5, 8, 9, 10,
    11, 12, 13, 14, 15, 15, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 25, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 35, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 45, 47
], dtype=float)

DATA_MATRIX = numpy.expand_dims(ACCUMULATED_PRECIP_VALUES_WITH_NEG, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

THESE_DIM = (
    wrf_arw_utils.FORECAST_HOUR_DIM, wrf_arw_utils.ROW_DIM,
    wrf_arw_utils.COLUMN_DIM, wrf_arw_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    wrf_arw_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (wrf_arw_utils.ROW_DIM, wrf_arw_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    wrf_arw_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    wrf_arw_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

WRF_ARW_TABLE_WITH_NEG_PRECIP_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)


class WrfArwUtilsTests(unittest.TestCase):
    """Each method is a unit test for wrf_arw_utils.py."""

    def test_precip_from_incremental_to_full_run(self):
        """Ensures correct output from precip_from_incremental_to_full_run."""

        new_gfs_table_xarray = wrf_arw_utils.precip_from_incremental_to_full_run(
            WRF_ARW_TABLE_XARRAY
        )

        these_precip_values = (
            new_gfs_table_xarray[wrf_arw_utils.DATA_KEY].values[..., 0, 0, 0]
        )
        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES, atol=TOLERANCE
        ))

    def test_remove_negative_precip(self):
        """Ensures correct output from remove_negative_precip."""

        new_gfs_table_xarray = wrf_arw_utils.remove_negative_precip(
            WRF_ARW_TABLE_WITH_NEG_PRECIP_XARRAY
        )

        these_precip_values = (
            new_gfs_table_xarray[wrf_arw_utils.DATA_KEY].values[..., 0, 0, 0]
        )
        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_SANS_NEG,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
