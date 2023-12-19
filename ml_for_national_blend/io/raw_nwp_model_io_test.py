"""Unit tests for raw_nwp_model_io.py"""

import unittest
import numpy
from ml_for_national_blend.io import raw_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

DIRECTORY_NAME = 'foo'
MODEL_NAME = nwp_model_utils.WRF_ARW_MODEL_NAME
INIT_TIMES_UNIX_SEC = numpy.array([1667260800, 1667304000], dtype=int)
FORECAST_HOUR = 21
NWP_FORECAST_FILE_NAMES = [
    'foo/20221101/2230500000021', 'foo/20221101/2230512000021'
]


class RawNwpModelIoTests(unittest.TestCase):
    """Each method is a unit test for raw_nwp_model_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_file_name = raw_nwp_model_io.find_file(
                directory_name=DIRECTORY_NAME,
                model_name=MODEL_NAME,
                forecast_hour=FORECAST_HOUR,
                init_time_unix_sec=INIT_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == NWP_FORECAST_FILE_NAMES[i])

    def test_file_name_to_init_time(self):
        """Ensures correct output from file_name_to_init_time."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_time_unix_sec = raw_nwp_model_io.file_name_to_init_time(
                nwp_forecast_file_name=NWP_FORECAST_FILE_NAMES[i],
                model_name=MODEL_NAME
            )
            self.assertTrue(this_time_unix_sec == INIT_TIMES_UNIX_SEC[i])

    def test_file_name_to_forecast_hour(self):
        """Ensures correct output from file_name_to_forecast_hour."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_forecast_hour = raw_nwp_model_io.file_name_to_forecast_hour(
                NWP_FORECAST_FILE_NAMES[i]
            )
            self.assertTrue(this_forecast_hour == FORECAST_HOUR)


if __name__ == '__main__':
    unittest.main()
