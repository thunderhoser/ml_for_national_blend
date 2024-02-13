"""Unit tests for interp_nwp_model_io.py"""

import unittest
import numpy
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

DIRECTORY_NAME = 'foo'
MODEL_NAME = nwp_model_utils.WRF_ARW_MODEL_NAME
FORECAST_HOUR = 7
INIT_TIMES_UNIX_SEC = numpy.array([1667260800, 1667304000], dtype=int)
WRF_ARW_FILE_NAMES = [
    'foo/wrf_arw_2022-11-01-00_hour007.nc',
    'foo/wrf_arw_2022-11-01-12_hour007.nc'
]


class InterpNwpModelIoTests(unittest.TestCase):
    """Each method is a unit test for interp_nwp_model_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_file_name = interp_nwp_model_io.find_file(
                directory_name=DIRECTORY_NAME,
                model_name=MODEL_NAME,
                forecast_hour=FORECAST_HOUR,
                init_time_unix_sec=INIT_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == WRF_ARW_FILE_NAMES[i])

    def test_file_name_to_init_time(self):
        """Ensures correct output from file_name_to_init_time."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_time_unix_sec = interp_nwp_model_io.file_name_to_init_time(
                WRF_ARW_FILE_NAMES[i]
            )
            self.assertTrue(this_time_unix_sec == INIT_TIMES_UNIX_SEC[i])

    def test_file_name_to_model_name(self):
        """Ensures correct output from file_name_to_model_name."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_model_name = interp_nwp_model_io.file_name_to_model_name(
                WRF_ARW_FILE_NAMES[i]
            )
            self.assertTrue(this_model_name == MODEL_NAME)

    def test_file_name_to_forecast_hour(self):
        """Ensures correct output from file_name_to_forecast_hour."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_forecast_hour = interp_nwp_model_io.file_name_to_forecast_hour(
                WRF_ARW_FILE_NAMES[i]
            )
            self.assertTrue(this_forecast_hour == FORECAST_HOUR)


if __name__ == '__main__':
    unittest.main()
