"""Unit tests for urma_io.py"""

import unittest
from ml_for_national_blend.io import urma_io

DIRECTORY_NAME = 'foo'
VALID_DATE_STRINGS = ['20221101', '20200229']
URMA_FILE_NAMES = ['foo/urma_20221101.nc', 'foo/urma_20200229.nc']


class UrmaIoTests(unittest.TestCase):
    """Each method is a unit test for urma_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        for i in range(len(VALID_DATE_STRINGS)):
            this_file_name = urma_io.find_file(
                directory_name=DIRECTORY_NAME,
                valid_date_string=VALID_DATE_STRINGS[i],
                raise_error_if_missing=False
            )
            self.assertTrue(this_file_name == URMA_FILE_NAMES[i])

    def test_file_name_to_date(self):
        """Ensures correct output from file_name_to_date."""

        for i in range(len(VALID_DATE_STRINGS)):
            this_date_string = urma_io.file_name_to_date(URMA_FILE_NAMES[i])
            self.assertTrue(this_date_string == VALID_DATE_STRINGS[i])


if __name__ == '__main__':
    unittest.main()
