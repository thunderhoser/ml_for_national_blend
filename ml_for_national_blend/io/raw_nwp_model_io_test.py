"""Unit tests for raw_nwp_model_io.py"""

import unittest
import numpy
from ml_for_national_blend.io import raw_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

DIRECTORY_NAME = 'foo'
DEFAULT_MODEL_NAME = nwp_model_utils.WRF_ARW_MODEL_NAME
INIT_TIMES_UNIX_SEC = numpy.array([1667260800, 1667304000], dtype=int)
FORECAST_HOUR = 21

DEFAULT_FORECAST_FILE_NAMES = [
    'foo/20221101/2230500000021', 'foo/20221101/2230512000021'
]
GLAMP_FORECAST_FILE_NAMES = [
    'foo/20221101/23305000021', 'foo/20221101/23305120021'
]


class RawNwpModelIoTests(unittest.TestCase):
    """Each method is a unit test for raw_nwp_model_io.py."""

    def test_find_file_default(self):
        """Ensures correct output from find_file.

        In this case, assuming default NWP model (not gridded LAMP).
        """

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_file_name = raw_nwp_model_io.find_file(
                directory_name=DIRECTORY_NAME,
                model_name=DEFAULT_MODEL_NAME,
                forecast_hour=FORECAST_HOUR,
                init_time_unix_sec=INIT_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == DEFAULT_FORECAST_FILE_NAMES[i])

    def test_find_file_glamp(self):
        """Ensures correct output from find_file.

        In this case, assuming NWP model = gridded LAMP.
        """

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_file_name = raw_nwp_model_io.find_file(
                directory_name=DIRECTORY_NAME,
                model_name=nwp_model_utils.GRIDDED_LAMP_MODEL_NAME,
                forecast_hour=FORECAST_HOUR,
                init_time_unix_sec=INIT_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == GLAMP_FORECAST_FILE_NAMES[i])

    def test_file_name_to_init_time_default(self):
        """Ensures correct output from file_name_to_init_time.

        In this case, assuming default NWP model (not gridded LAMP).
        """

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_time_unix_sec = raw_nwp_model_io.file_name_to_init_time(
                nwp_forecast_file_name=DEFAULT_FORECAST_FILE_NAMES[i],
                model_name=DEFAULT_MODEL_NAME
            )
            self.assertTrue(this_time_unix_sec == INIT_TIMES_UNIX_SEC[i])

    def test_file_name_to_init_time_glamp(self):
        """Ensures correct output from file_name_to_init_time.

        In this case, assuming NWP model = gridded LAMP.
        """

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_time_unix_sec = raw_nwp_model_io.file_name_to_init_time(
                nwp_forecast_file_name=GLAMP_FORECAST_FILE_NAMES[i],
                model_name=nwp_model_utils.GRIDDED_LAMP_MODEL_NAME
            )
            self.assertTrue(this_time_unix_sec == INIT_TIMES_UNIX_SEC[i])

    def test_file_name_to_forecast_hour_default(self):
        """Ensures correct output from file_name_to_forecast_hour.

        In this case, assuming default NWP model (not gridded LAMP).
        """

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_forecast_hour = raw_nwp_model_io.file_name_to_forecast_hour(
                DEFAULT_FORECAST_FILE_NAMES[i]
            )
            self.assertTrue(this_forecast_hour == FORECAST_HOUR)

    def test_file_name_to_forecast_hour_glamp(self):
        """Ensures correct output from file_name_to_forecast_hour.

        In this case, assuming NWP model = gridded LAMP.
        """

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_forecast_hour = raw_nwp_model_io.file_name_to_forecast_hour(
                GLAMP_FORECAST_FILE_NAMES[i]
            )
            self.assertTrue(this_forecast_hour == FORECAST_HOUR)


if __name__ == '__main__':
    unittest.main()
