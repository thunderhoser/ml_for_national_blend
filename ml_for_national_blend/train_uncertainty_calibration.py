"""Trains uncertainty-calibration model to bias-correct ensemble spread."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
import bias_clustering
import bias_correction

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'

# TODO(thunderhoser): Allow multiple input directories, one per lead time.
INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings'
CLUSTER_FILE_ARG_NAME = 'input_cluster_file_name'
TARGET_FIELD_ARG_NAME = 'target_field_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing non-bias-corrected predictions.  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
INIT_TIME_LIMITS_HELP_STRING = (
    'List of two initialization times, specifying the beginning and end of the '
    'training period.  Time format is "yyyy-mm-dd-HH".'
)
CLUSTER_FILE_HELP_STRING = (
    'Path to cluster file, from which spatial clusters will be read by '
    '`bias_clustering.read_file`.  If you want to train one model for the '
    'whole spatial domain, rather than one per spatial cluster, leave this '
    'argument alone.'
)
TARGET_FIELD_HELP_STRING = (
    'Target field for which to train UC model.  Field name must be accepted by '
    '`urma_utils.check_field_name`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  The suite of trained UC models will be saved here, '
    'in Dill format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_TIME_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLUSTER_FILE_ARG_NAME, type=str, required=True, default='',
    help=CLUSTER_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELD_ARG_NAME, type=str, required=True,
    help=TARGET_FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_time_limit_strings, cluster_file_name,
         target_field_name, output_file_name):
    """Trains isotonic-regression model to bias-correct ensemble mean.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of this script.
    :param init_time_limit_strings: Same.
    :param cluster_file_name: Same.
    :param target_field_name: Same.
    :param output_file_name: Same.
    """

    if cluster_file_name == '':
        cluster_table_xarray = None
    else:
        print('Reading data from: "{0:s}"...'.format(cluster_file_name))
        cluster_table_xarray = bias_clustering.read_file(cluster_file_name)

    init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings
    ], dtype=int)

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_time_unix_sec=init_time_limits_unix_sec[0],
        last_init_time_unix_sec=init_time_limits_unix_sec[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    num_files = len(prediction_file_names)
    prediction_tables_xarray = [xarray.Dataset()] * num_files
    do_iso_reg_before_uncertainty_calib = None

    for k in range(num_files):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[k]))
        prediction_tables_xarray[k] = prediction_io.read_file(
            prediction_file_names[k]
        )

        field_index = numpy.where(
            prediction_tables_xarray[k][prediction_io.FIELD_NAME_KEY] ==
            target_field_name
        )[0][0]

        prediction_tables_xarray[k] = prediction_tables_xarray[k].isel(
            {prediction_io.FIELD_DIM: numpy.array([field_index], dtype=int)}
        )

        ptx_k = prediction_tables_xarray[k]

        if do_iso_reg_before_uncertainty_calib is None:
            do_iso_reg_before_uncertainty_calib = (
                ptx_k.attrs[prediction_io.ISOTONIC_MODEL_FILES_KEY] is not None
            )

        assert (
            do_iso_reg_before_uncertainty_calib ==
            (ptx_k.attrs[prediction_io.ISOTONIC_MODEL_FILES_KEY] is not None)
        )
        assert (
            ptx_k.attrs[prediction_io.UNCERTAINTY_CALIB_MODEL_FILES_KEY] is None
        )

        prediction_tables_xarray[k] = (
            prediction_io.prep_for_uncertainty_calib_training(
                prediction_tables_xarray[k]
            )
        )

    print(SEPARATOR_STRING)

    model_dict = bias_correction.train_model_suite(
        prediction_tables_xarray=prediction_tables_xarray,
        cluster_table_xarray=cluster_table_xarray,
        target_field_name=target_field_name,
        do_uncertainty_calibration=True,
        do_iso_reg_before_uncertainty_calib=do_iso_reg_before_uncertainty_calib
    )
    print(SEPARATOR_STRING)

    print('Writing model suite to: "{0:s}"...'.format(output_file_name))
    bias_correction.write_file(
        dill_file_name=output_file_name, model_dict=model_dict
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_TIME_LIMITS_ARG_NAME
        ),
        cluster_file_name=getattr(INPUT_ARG_OBJECT, CLUSTER_FILE_ARG_NAME),
        target_field_name=getattr(INPUT_ARG_OBJECT, TARGET_FIELD_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
