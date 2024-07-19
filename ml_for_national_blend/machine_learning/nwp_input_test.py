"""Unit tests for nwp_input.py"""

import unittest
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.machine_learning import nwp_input

TOLERANCE = 1e-6

# The following constants are used to test _find_best_lead_times.
NWP_DIRECTORY_NAME = 'foo'
INIT_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2024-07-19-12', '%Y-%m-%d-%H'
)
NWP_MODEL_NAME = nwp_model_utils.GFS_MODEL_NAME
DESIRED_FORECAST_HOUR = 75

FIRST_FORECAST_HOURS = numpy.array([6, 12, 24, 48, 72, 78, 96], dtype=int)
FIRST_NWP_FILE_NAMES = [
    interp_nwp_model_io.find_file(
        directory_name=NWP_DIRECTORY_NAME,
        init_time_unix_sec=INIT_TIME_UNIX_SEC,
        forecast_hour=f,
        model_name=NWP_MODEL_NAME,
        raise_error_if_missing=False
    )
    for f in FIRST_FORECAST_HOURS
]
FIRST_DESIRED_NWP_FILE_NAMES = [
    interp_nwp_model_io.find_file(
        directory_name=NWP_DIRECTORY_NAME,
        init_time_unix_sec=INIT_TIME_UNIX_SEC,
        forecast_hour=f,
        model_name=NWP_MODEL_NAME,
        raise_error_if_missing=False
    )
    for f in [72, 78]
]

SECOND_FORECAST_HOURS = numpy.array([6, 12, 24, 48, 72, 96], dtype=int)
SECOND_NWP_FILE_NAMES = [
    interp_nwp_model_io.find_file(
        directory_name=NWP_DIRECTORY_NAME,
        init_time_unix_sec=INIT_TIME_UNIX_SEC,
        forecast_hour=f,
        model_name=NWP_MODEL_NAME,
        raise_error_if_missing=False
    )
    for f in SECOND_FORECAST_HOURS
]
SECOND_DESIRED_NWP_FILE_NAMES = [
    interp_nwp_model_io.find_file(
        directory_name=NWP_DIRECTORY_NAME,
        init_time_unix_sec=INIT_TIME_UNIX_SEC,
        forecast_hour=f,
        model_name=NWP_MODEL_NAME,
        raise_error_if_missing=False
    )
    for f in [72, 96]
]

THIRD_FORECAST_HOURS = numpy.array([6, 12, 24, 48, 96], dtype=int)
THIRD_NWP_FILE_NAMES = [
    interp_nwp_model_io.find_file(
        directory_name=NWP_DIRECTORY_NAME,
        init_time_unix_sec=INIT_TIME_UNIX_SEC,
        forecast_hour=f,
        model_name=NWP_MODEL_NAME,
        raise_error_if_missing=False
    )
    for f in THIRD_FORECAST_HOURS
]
THIRD_DESIRED_NWP_FILE_NAMES = []

# The following constants are used to test _interp_predictors_by_lead_time.
TARGET_LEAD_TIMES_HOURS = numpy.array([12, 24, 41, 48], dtype=int)
SOURCE_LEAD_TIMES_HOURS = numpy.array([18, 27, 69], dtype=int)

THIS_FIRST_MATRIX = numpy.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=float)

THIS_SECOND_MATRIX = numpy.array([
    [7, 8, 9],
    [10, 11, 12]
], dtype=float)

THIS_THIRD_MATRIX = numpy.array([
    [13, 14, 15],
    [16, 17, 18]
], dtype=float)

SOURCE_PREDICTOR_MATRIX = numpy.stack(
    [THIS_FIRST_MATRIX, THIS_SECOND_MATRIX, THIS_THIRD_MATRIX], axis=-1
)
SOURCE_PREDICTOR_MATRIX = numpy.expand_dims(SOURCE_PREDICTOR_MATRIX, axis=-1)

THIS_FIRST_MATRIX = numpy.full((2, 3), numpy.nan)

THIS_SECOND_MATRIX = numpy.array([
    [5, 6, 7],
    [8, 9, 10]
], dtype=float)

THIS_THIRD_MATRIX = numpy.array([
    [9, 10, 11],
    [12, 13, 14]
], dtype=float)

THIS_FOURTH_MATRIX = numpy.array([
    [10, 11, 12],
    [13, 14, 15]
], dtype=float)

TARGET_PREDICTOR_MATRIX = numpy.stack([
    THIS_FIRST_MATRIX, THIS_SECOND_MATRIX,
    THIS_THIRD_MATRIX, THIS_FOURTH_MATRIX
], axis=-1)

TARGET_PREDICTOR_MATRIX = numpy.expand_dims(TARGET_PREDICTOR_MATRIX, axis=-1)


class NwpInputTests(unittest.TestCase):
    """Each method is a unit test for nwp_input.py."""

    def test_find_best_lead_times_first(self):
        """Ensures correct output from _find_best_lead_times.

        With first set of inputs.
        """

        these_file_names = nwp_input._find_best_lead_times(
            nwp_forecast_file_names=FIRST_NWP_FILE_NAMES,
            desired_lead_time_hours=DESIRED_FORECAST_HOUR
        )
        self.assertTrue(these_file_names == FIRST_DESIRED_NWP_FILE_NAMES)

    def test_find_best_lead_times_second(self):
        """Ensures correct output from _find_best_lead_times.

        With second set of inputs.
        """

        these_file_names = nwp_input._find_best_lead_times(
            nwp_forecast_file_names=SECOND_NWP_FILE_NAMES,
            desired_lead_time_hours=DESIRED_FORECAST_HOUR
        )
        self.assertTrue(these_file_names == SECOND_DESIRED_NWP_FILE_NAMES)

    def test_find_best_lead_times_third(self):
        """Ensures correct output from _find_best_lead_times.

        With third set of inputs.
        """

        these_file_names = nwp_input._find_best_lead_times(
            nwp_forecast_file_names=THIRD_NWP_FILE_NAMES,
            desired_lead_time_hours=DESIRED_FORECAST_HOUR
        )
        self.assertTrue(these_file_names == THIRD_DESIRED_NWP_FILE_NAMES)

    def test_interp_predictors_by_lead_time(self):
        """Ensures correct output from _interp_predictors_by_lead_time."""

        this_predictor_matrix = nwp_input._interp_predictors_by_lead_time(
            predictor_matrix=SOURCE_PREDICTOR_MATRIX,
            source_lead_times_hours=SOURCE_LEAD_TIMES_HOURS,
            target_lead_times_hours=TARGET_LEAD_TIMES_HOURS
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, TARGET_PREDICTOR_MATRIX,
            atol=TOLERANCE, equal_nan=True
        ))


if __name__ == '__main__':
    unittest.main()
