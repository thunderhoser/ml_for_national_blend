"""Unit tests for wrf_arw_io.py"""

import unittest
import numpy
from ml_for_national_blend.io import wrf_arw_io

DIRECTORY_NAME = 'foo'
INIT_TIMES_UNIX_SEC = numpy.array([1667260800, 1667304000], dtype=int)
WRF_ARW_FILE_NAMES = [
    'foo/wrf_arw_2022-11-01-00.nc', 'foo/wrf_arw_2022-11-01-12.nc'
]


class WrfArwIoTests(unittest.TestCase):
    """Each method is a unit test for wrf_arw_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_file_name = wrf_arw_io.find_file(
                directory_name=DIRECTORY_NAME,
                init_time_unix_sec=INIT_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == WRF_ARW_FILE_NAMES[i])

    def test_file_name_to_init_time(self):
        """Ensures correct output from file_name_to_init_time."""

        for i in range(len(INIT_TIMES_UNIX_SEC)):
            this_time_unix_sec = wrf_arw_io.file_name_to_init_time(
                WRF_ARW_FILE_NAMES[i]
            )
            self.assertTrue(this_time_unix_sec == INIT_TIMES_UNIX_SEC[i])


if __name__ == '__main__':
    unittest.main()
