"""Unit tests for nwp_model_utils.py"""

import unittest
import numpy
import xarray
from ml_for_national_blend.utils import nwp_model_utils

TOLERANCE = 1e-6

# The following constants are used to test precip_from_incremental_to_full_run.
INCREMENTAL_PRECIP_VALUES_WRF_ARW = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    5, 6, 7, 8, 9
], dtype=float)

ACCUMULATED_PRECIP_VALUES_WRF_ARW = numpy.array([
    0,
    1, 3, 4, 6, 7, 9, 10, 12, 13, 15,
    16, 18, 19, 21, 22, 24, 25, 27, 28, 30,
    31, 33, 34, 36, 37, 39, 40, 42, 43, 45,
    48, 52, 55, 59, 62, 66, 69, 73, 76, 80, 83, 87,
    92, 98, 105, 113, 122
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES_WRF_ARW, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM: nwp_model_utils.model_to_forecast_hours(
        model_name=nwp_model_utils.WRF_ARW_MODEL_NAME, init_time_unix_sec=0
    ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

WRF_ARW_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

INCREMENTAL_PRECIP_VALUES_NAM = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1
], dtype=float)

ACCUMULATED_PRECIP_VALUES_NAM = numpy.array([
    0,
    1, 3, 4, 6, 7, 9, 10, 12, 13, 15,
    16
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES_NAM, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM: nwp_model_utils.model_to_forecast_hours(
        model_name=nwp_model_utils.NAM_MODEL_NAME, init_time_unix_sec=0
    ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

NAM_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

INCREMENTAL_PRECIP_VALUES_NAM_NEST = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    5, 6, 7, 8, 9,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4
], dtype=float)

ACCUMULATED_PRECIP_VALUES_NAM_NEST = numpy.array([
    0,
    1, 2, 3, 4, 3, 5, 4, 5, 6, 7,
    6, 8, 7, 8, 9, 10, 9, 11, 10, 11,
    12, 13, 12, 14, 13, 14, 15, 16, 15, 17,
    18, 19, 22, 23, 22, 26, 25, 26, 29, 30, 29, 33,
    34, 35, 42, 43, 44,
    47, 48, 47, 51, 50, 51, 54, 55, 54, 58, 57, 58
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES_NAM_NEST, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM: nwp_model_utils.model_to_forecast_hours(
        model_name=nwp_model_utils.NAM_NEST_MODEL_NAME, init_time_unix_sec=0
    ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

NAM_NEST_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

INCREMENTAL_PRECIP_VALUES_GEFS = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    5, 6, 7, 8, 9,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8,
    9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10
], dtype=float)

ACCUMULATED_PRECIP_VALUES_GEFS = numpy.array([
    0,
    1, 3, 2, 4, 3, 5, 4, 6, 5, 7,
    6, 8, 7, 9, 8, 10, 9, 11, 10, 12,
    11, 13, 12, 14, 13, 15, 14, 16, 15, 17,
    18, 22, 21, 25, 24, 28, 27, 31, 30, 34, 33, 37,
    38, 44, 45, 53, 54,
    59, 60, 65, 66, 71, 72, 77, 78, 83, 84, 89, 90,
    95, 96, 101, 102, 107, 108, 113, 114, 119, 120,
    125, 126, 131, 132, 137, 138, 143, 144, 149, 150,
    157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240,
    249, 259, 268, 278, 287, 297, 306, 316, 325, 335, 344, 354
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES_GEFS, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM: nwp_model_utils.model_to_forecast_hours(
        model_name=nwp_model_utils.GEFS_MODEL_NAME, init_time_unix_sec=0
    ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

GEFS_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

# The following constants are used to test
# old_gfs_or_gefs_precip_from_incr_to_full.
INCREMENTAL_PRECIP_VALUES_OLD_GEFS = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    5, 6, 7, 8, 9,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    5, 6, 5, 6
], dtype=float)

ACCUMULATED_PRECIP_VALUES_OLD_GEFS = numpy.array([
    0,
    1, 3, 4, 6, 7, 9, 10, 12, 13, 15,
    16, 18, 19, 21, 22, 24, 25, 27, 28, 30,
    31, 33, 34, 36, 37, 39, 40, 42, 43, 45,
    48, 52, 55, 59, 62, 66, 69, 73, 76, 80, 83, 87,
    92, 98, 105, 113, 122,
    127, 133, 138, 144, 149, 155, 160, 166, 171, 177, 182, 188,
    193, 199, 204, 210
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES_OLD_GEFS, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM:
        nwp_model_utils.model_to_old_forecast_hours(
            nwp_model_utils.GEFS_MODEL_NAME
        ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

OLD_GEFS_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

INCREMENTAL_PRECIP_VALUES_OLD_GFS = numpy.array([
    0,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
    5, 6, 7, 8, 9,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    5, 6, 5, 6, 5, 6, 5, 6, 5, 6,
    7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8
], dtype=float)

ACCUMULATED_PRECIP_VALUES_OLD_GFS = numpy.array([
    0,
    1, 3, 2, 4, 3, 5, 4, 6, 5, 7,
    6, 8, 7, 9, 8, 10, 9, 11, 10, 12,
    11, 13, 12, 14, 13, 15, 14, 16, 15, 17,
    18, 22, 21, 25, 24, 28, 27, 31, 30, 34, 33, 37,
    38, 44, 45, 53, 54,
    59, 60, 65, 66, 71, 72, 77, 78, 83, 84, 89, 90,
    95, 96, 101, 102, 107, 108, 113, 114, 119, 120,
    125, 126, 131, 132, 137, 138, 143, 144, 149, 150,
    157, 165, 172, 180, 187, 195, 202, 210, 217, 225, 232, 240
], dtype=float)

DATA_MATRIX = numpy.expand_dims(INCREMENTAL_PRECIP_VALUES_OLD_GFS, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM:
        nwp_model_utils.model_to_old_forecast_hours(
            nwp_model_utils.GFS_MODEL_NAME
        ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

OLD_GFS_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

# The following constants are used to test remove_negative_precip.
ACCUMULATED_PRECIP_VALUES_WRF_ARW_WITH_NEG = numpy.array([
    numpy.nan,
    1, 2, 3, 4, 5, 4, numpy.nan, 8, 9, 10,
    11, 12, 13, 14, 15, 13, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 22, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 31, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 40, 47
], dtype=float)

ACCUMULATED_PRECIP_VALUES_WRF_ARW_SANS_NEG = numpy.array([
    numpy.nan,
    1, 2, 3, 4, 5, 5, 5, 8, 9, 10,
    11, 12, 13, 14, 15, 15, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 25, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 35, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 45, 47
], dtype=float)

DATA_MATRIX = numpy.expand_dims(
    ACCUMULATED_PRECIP_VALUES_WRF_ARW_WITH_NEG, axis=-1
)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)
DATA_MATRIX = numpy.expand_dims(DATA_MATRIX, axis=-1)

THESE_DIM = (
    nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
    nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
)
MAIN_DATA_DICT = {
    nwp_model_utils.DATA_KEY: (THESE_DIM, DATA_MATRIX)
}

THESE_DIM = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
MAIN_DATA_DICT.update({
    nwp_model_utils.LATITUDE_KEY: (THESE_DIM, numpy.array([[40.02]])),
    nwp_model_utils.LONGITUDE_KEY: (THESE_DIM, numpy.array([[254.75]]))
})

COORD_DICT = {
    nwp_model_utils.FORECAST_HOUR_DIM: nwp_model_utils.model_to_forecast_hours(
        model_name=nwp_model_utils.WRF_ARW_MODEL_NAME, init_time_unix_sec=0
    ),
    nwp_model_utils.ROW_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.COLUMN_DIM: numpy.array([0], dtype=int),
    nwp_model_utils.FIELD_DIM: [nwp_model_utils.PRECIP_NAME]
}

NWP_TABLE_WITH_NEG_PRECIP_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)


class NwpModelUtilsTests(unittest.TestCase):
    """Each method is a unit test for nwp_model_utils.py."""

    def test_precip_from_incremental_to_full_run_wrf_arw(self):
        """Ensures correct output from precip_from_incremental_to_full_run.

        In this case, assuming the model is WRF-ARW.
        """

        new_forecast_table_xarray = (
            nwp_model_utils.precip_from_incremental_to_full_run(
                nwp_forecast_table_xarray=WRF_ARW_TABLE_XARRAY,
                model_name=nwp_model_utils.WRF_ARW_MODEL_NAME,
                init_time_unix_sec=0
            )
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_WRF_ARW,
            atol=TOLERANCE
        ))

    def test_precip_from_incremental_to_full_run_nam(self):
        """Ensures correct output from precip_from_incremental_to_full_run.

        In this case, assuming the model is NAM.
        """

        new_forecast_table_xarray = (
            nwp_model_utils.precip_from_incremental_to_full_run(
                nwp_forecast_table_xarray=NAM_TABLE_XARRAY,
                model_name=nwp_model_utils.NAM_MODEL_NAME,
                init_time_unix_sec=0
            )
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_NAM,
            atol=TOLERANCE
        ))

    def test_precip_from_incremental_to_full_run_nam_nest(self):
        """Ensures correct output from precip_from_incremental_to_full_run.

        In this case, assuming the model is NAM Nest.
        """

        new_forecast_table_xarray = (
            nwp_model_utils.precip_from_incremental_to_full_run(
                nwp_forecast_table_xarray=NAM_NEST_TABLE_XARRAY,
                model_name=nwp_model_utils.NAM_NEST_MODEL_NAME,
                init_time_unix_sec=0
            )
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_NAM_NEST,
            atol=TOLERANCE
        ))

    def test_precip_from_incremental_to_full_run_gefs(self):
        """Ensures correct output from precip_from_incremental_to_full_run.

        In this case, assuming the model is GEFS.
        """

        new_forecast_table_xarray = (
            nwp_model_utils.precip_from_incremental_to_full_run(
                nwp_forecast_table_xarray=GEFS_TABLE_XARRAY,
                model_name=nwp_model_utils.GEFS_MODEL_NAME,
                init_time_unix_sec=0
            )
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_GEFS,
            atol=TOLERANCE
        ))

    def test_old_gfs_or_gefs_precip_from_incr_to_full_gefs(self):
        """Ensures correctness of old_gfs_or_gefs_precip_from_incr_to_full.

        In this case, assuming the model is GEFS.
        """

        new_forecast_table_xarray = (
            nwp_model_utils.old_gfs_or_gefs_precip_from_incr_to_full(
                nwp_forecast_table_xarray=OLD_GEFS_TABLE_XARRAY,
                model_name=nwp_model_utils.GEFS_MODEL_NAME
            )
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_OLD_GEFS,
            atol=TOLERANCE
        ))

    def test_old_gfs_or_gefs_precip_from_incr_to_full_gfs(self):
        """Ensures correctness of old_gfs_or_gefs_precip_from_incr_to_full.

        In this case, assuming the model is GFS.
        """

        new_forecast_table_xarray = (
            nwp_model_utils.old_gfs_or_gefs_precip_from_incr_to_full(
                nwp_forecast_table_xarray=OLD_GFS_TABLE_XARRAY,
                model_name=nwp_model_utils.GFS_MODEL_NAME
            )
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_OLD_GFS,
            atol=TOLERANCE
        ))

    def test_remove_negative_precip(self):
        """Ensures correct output from remove_negative_precip."""

        new_forecast_table_xarray = nwp_model_utils.remove_negative_precip(
            NWP_TABLE_WITH_NEG_PRECIP_XARRAY
        )

        these_precip_values = new_forecast_table_xarray[
            nwp_model_utils.DATA_KEY
        ].values[..., 0, 0, 0]

        self.assertTrue(numpy.allclose(
            these_precip_values, ACCUMULATED_PRECIP_VALUES_WRF_ARW_SANS_NEG,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
