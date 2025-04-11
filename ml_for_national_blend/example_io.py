"""Input/output methods for fully processed NN examples."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking
import neural_net_training_simple as nn_training_simple

TIME_FORMAT = '%Y-%m-%d-%H'
SYNOPTIC_TIME_INTERVAL_SEC = 6 * 3600


def find_file(directory_name, init_time_unix_sec, raise_error_if_missing=True):
    """Finds .npz file with fully processed NN examples.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: example_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_equals(
        numpy.mod(init_time_unix_sec, SYNOPTIC_TIME_INTERVAL_SEC), 0
    )
    error_checking.assert_is_boolean(raise_error_if_missing)

    example_file_name = '{0:s}/learning_example_{1:s}.npz'.format(
        directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )

    if os.path.isfile(example_file_name) or not raise_error_if_missing:
        return example_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        example_file_name
    )
    raise ValueError(error_string)


def read_file(npz_file_name):
    """Reads fully processed NN examples from .npz file.

    :param npz_file_name: Path to input file.
    :return: predictor_matrices: See output doc for
        `neural_net_training_simple.create_data`.
    :return: target_matrix: Same.
    """

    error_checking.assert_file_exists(npz_file_name)
    this_dict = numpy.load(npz_file_name, allow_pickle=True)

    return (
        tuple(this_dict['predictor_matrices'].tolist()),
        this_dict['target_matrix']
    )


def write_file(data_dict, npz_file_name):
    """Writes fully processed NN examples to .npz file.

    :param data_dict: Dictionary returned by
        `neural_net_training_simple.create_data`.
    :param npz_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=npz_file_name)

    output_dict = {
        'predictor_matrix{0:d}'.format(i): this_array
        for i, this_array in
        enumerate(data_dict[nn_training_simple.PREDICTOR_MATRICES_KEY])
    }
    output_dict['target_matrix'] = data_dict[
        nn_training_simple.TARGET_MATRIX_KEY
    ]

    numpy.savez(npz_file_name, **output_dict)
