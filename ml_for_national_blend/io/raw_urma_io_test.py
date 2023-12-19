"""Unit tests for raw_urma_io.py"""

import unittest
import numpy
from ml_for_national_blend.io import raw_urma_io

DIRECTORY_NAME = 'foo'
VALID_TIMES_UNIX_SEC = numpy.array([1667264400, 1667307600], dtype=int)
URMA_FILE_NAMES = [
    'foo/20221101/urma2p5.t01z.2dvaranl_ndfd.grb2_wexp',
    'foo/20221101/urma2p5.t13z.2dvaranl_ndfd.grb2_wexp'
]


class RawUrmaIoTests(unittest.TestCase):
    """Each method is a unit test for raw_urma_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        for i in range(len(VALID_TIMES_UNIX_SEC)):
            this_file_name = raw_urma_io.find_file(
                directory_name=DIRECTORY_NAME,
                valid_time_unix_sec=VALID_TIMES_UNIX_SEC[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == URMA_FILE_NAMES[i])

    def test_file_name_to_valid_time(self):
        """Ensures correct output from file_name_to_valid_time."""

        for i in range(len(VALID_TIMES_UNIX_SEC)):
            this_time_unix_sec = raw_urma_io.file_name_to_valid_time(
                urma_file_name=URMA_FILE_NAMES[i]
            )
            self.assertTrue(this_time_unix_sec == VALID_TIMES_UNIX_SEC[i])


if __name__ == '__main__':
    unittest.main()
