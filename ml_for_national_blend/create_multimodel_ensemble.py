"""Blends predictions from several models into ensemble."""

import os
import sys
import copy
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import error_checking
import prediction_io

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H'
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600

MODEL_FILE_KEY = prediction_io.MODEL_FILE_KEY
ISOTONIC_MODEL_FILE_KEY = prediction_io.ISOTONIC_MODEL_FILES_KEY
UNCERTAINTY_CALIB_MODEL_FILE_KEY = (
    prediction_io.UNCERTAINTY_CALIB_MODEL_FILES_KEY
)

INPUT_DIRS_ARG_NAME = 'input_prediction_dir_names'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_total_ensemble_size'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_DIRS_HELP_STRING = (
    'List of paths to input directories, one per model.  Within each '
    'directory, relevant files will be found by `prediction_io.find_file` and '
    'read by `prediction_io.read_file`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'This script will create an ensemble for every forecast-initialization '
    'time in the period `{0:s}`...`{1:s}`.  Time format should be '
    'yyyy-mm-dd-HH.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = FIRST_INIT_TIME_HELP_STRING
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    'Max total ensemble size.  Letting S = `{0:s}`, if the total ensemble size '
    'after blending all models is > S, then S members will be randomly chosen.'
).format(
    MAX_ENSEMBLE_SIZE_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  For every forecast-init time, the multi-model '
    'ensemble will be written here by `prediction_io.write_file`, to an exact '
    'location determined by `prediction_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=False,
    default=int(1e10), help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _create_ensemble_one_init_time(
        input_prediction_dir_names, max_total_ensemble_size,
        init_time_unix_sec, output_prediction_dir_name):
    """Creates multi-model ensemble for one forecast-init time.

    :param input_prediction_dir_names: See documentation at top of this script.
    :param max_total_ensemble_size: Same.
    :param init_time_unix_sec: Same.
    :param output_prediction_dir_name: Same.
    """

    num_models = len(input_prediction_dir_names)
    prediction_tables_xarray = [xarray.Dataset()] * num_models
    target_field_names = []
    latitude_matrix_deg_n = numpy.array([])
    longitude_matrix_deg_e = numpy.array([])
    target_matrix = numpy.array([])
    prediction_matrix = numpy.array([])
    model_file_names = [''] * num_models

    for i in range(num_models):
        this_prediction_file_name = prediction_io.find_file(
            directory_name=input_prediction_dir_names[i],
            init_time_unix_sec=init_time_unix_sec,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_prediction_file_name):
            return

        print('Reading data from: "{0:s}"...'.format(this_prediction_file_name))
        prediction_tables_xarray[i] = prediction_io.read_file(
            this_prediction_file_name
        )

        ptx_i = prediction_tables_xarray[i]
        model_file_names[i] = ptx_i.attrs[prediction_io.MODEL_FILE_KEY]
        these_target_field_names = ptx_i[prediction_io.FIELD_NAME_KEY].values.tolist()
        this_latitude_matrix_deg_n = ptx_i[prediction_io.LATITUDE_KEY].values
        this_longitude_matrix_deg_e = ptx_i[prediction_io.LONGITUDE_KEY].values
        this_target_matrix = ptx_i[prediction_io.TARGET_KEY].values

        if i == 0:
            target_field_names = copy.deepcopy(these_target_field_names)
            latitude_matrix_deg_n = this_latitude_matrix_deg_n + 0.
            longitude_matrix_deg_e = this_longitude_matrix_deg_e + 0.
            target_matrix = this_target_matrix + 0.

        assert target_field_names == these_target_field_names
        assert numpy.allclose(
            latitude_matrix_deg_n, this_latitude_matrix_deg_n, atol=TOLERANCE
        )
        assert numpy.allclose(
            longitude_matrix_deg_e, this_longitude_matrix_deg_e, atol=TOLERANCE
        )
        assert numpy.allclose(target_matrix, this_target_matrix, atol=TOLERANCE)

        if i == 0:
            prediction_matrix = ptx_i[prediction_io.PREDICTION_KEY].values + 0.
        else:
            prediction_matrix = numpy.concatenate(
                [prediction_matrix, ptx_i[prediction_io.PREDICTION_KEY].values],
                axis=-1
            )

    ensemble_size = prediction_matrix.shape[-1]
    if ensemble_size > max_total_ensemble_size:
        member_indices = numpy.linspace(
            0, ensemble_size - 1, num=ensemble_size, dtype=int
        )
        member_indices = numpy.random.choice(
            member_indices, size=max_total_ensemble_size, replace=False
        )
        prediction_matrix = prediction_matrix[..., member_indices]

    output_file_name = prediction_io.find_file(
        directory_name=output_prediction_dir_name,
        init_time_unix_sec=init_time_unix_sec,
        raise_error_if_missing=False
    )

    print('Writing multi-model ensemble to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=target_matrix,
        prediction_matrix=prediction_matrix,
        latitude_matrix_deg_n=latitude_matrix_deg_n,
        longitude_matrix_deg_e=longitude_matrix_deg_e,
        field_names=target_field_names,
        init_time_unix_sec=init_time_unix_sec,
        model_file_name=' '.join(model_file_names),
        isotonic_model_file_names=None,
        uncertainty_calib_model_file_names=None
    )


def _run(input_prediction_dir_names, max_total_ensemble_size,
         first_init_time_string, last_init_time_string,
         output_prediction_dir_name):
    """Blends predictions from several models into ensemble.

    This is effectively the main method.

    :param input_prediction_dir_names: See documentation at top of this script.
    :param max_total_ensemble_size: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param output_prediction_dir_name: Same.
    """

    error_checking.assert_is_geq(max_total_ensemble_size, 2)

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=SYNOPTIC_TIME_INTERVAL_SEC,
        include_endpoint=True
    )

    for this_init_time_unix_sec in init_times_unix_sec:
        _create_ensemble_one_init_time(
            input_prediction_dir_names=input_prediction_dir_names,
            max_total_ensemble_size=max_total_ensemble_size,
            init_time_unix_sec=this_init_time_unix_sec,
            output_prediction_dir_name=output_prediction_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_dir_names=getattr(
            INPUT_ARG_OBJECT, INPUT_DIRS_ARG_NAME
        ),
        max_total_ensemble_size=getattr(
            INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME
        ),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        output_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME
        )
    )
