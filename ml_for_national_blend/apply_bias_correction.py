"""Applies bias correction -- inference time!"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
import bias_correction

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings'
ISOTONIC_FILES_ARG_NAME = 'isotonic_model_file_names'
UNCERTAINTY_CALIB_FILES_ARG_NAME = 'uncertainty_calib_model_file_names'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing non-bias-corrected predictions.  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
INIT_TIME_LIMITS_HELP_STRING = (
    'List of two initialization times, specifying the beginning and end of the '
    'period for which to apply bias correction.  Time format is '
    '"yyyy-mm-dd-HH".'
)
ISOTONIC_FILES_HELP_STRING = (
    'Paths to files with isotonic-regression models (one per target field), '
    'which will be used to bias-correct ensemble means.  If you do not want '
    'IR, leave this argument alone.'
)
UNCERTAINTY_CALIB_FILES_HELP_STRING = (
    'Paths to files with uncertainty-calibration modelsmodels (one per target '
    'field), which will be used to bias-correct ensemble spreads.  If you do '
    'not want UC, leave this argument alone.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Bias-corrected predictions will be written '
    'here by `prediction_io.write_file`, to exact locations determined by '
    '`prediction_io.find_file`.'
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
    '--' + ISOTONIC_FILES_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=ISOTONIC_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNCERTAINTY_CALIB_FILES_ARG_NAME, type=str, nargs='+',
    required=False, default=[''], help=UNCERTAINTY_CALIB_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, init_time_limit_strings, isotonic_model_file_names,
         uncertainty_calib_model_file_names, output_dir_name):
    """Applies bias correction -- inference time!

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param init_time_limit_strings: Same.
    :param isotonic_model_file_names: Same.
    :param uncertainty_calib_model_file_names: Same.
    :param output_dir_name: Same.
    """

    if (
            len(isotonic_model_file_names) == 1 and
            isotonic_model_file_names[0] == ''
    ):
        isotonic_model_file_names = None

    if (
            len(uncertainty_calib_model_file_names) == 1 and
            uncertainty_calib_model_file_names[0] == ''
    ):
        uncertainty_calib_model_file_names = None

    assert not (
        isotonic_model_file_names is None and
        uncertainty_calib_model_file_names is None
    )

    field_counts = []
    if isotonic_model_file_names is not None:
        field_counts.append(len(isotonic_model_file_names))
    if uncertainty_calib_model_file_names is not None:
        field_counts.append(len(uncertainty_calib_model_file_names))

    field_counts = numpy.array(field_counts, dtype=int)
    assert len(numpy.unique(field_counts)) == 1
    num_fields = field_counts[0]

    if isotonic_model_file_names is None:
        isotonic_model_dict_by_field = None
    else:
        isotonic_model_dict_by_field = [dict()] * num_fields

        for f in range(num_fields):
            print('Reading IR model from: "{0:s}"...'.format(
                isotonic_model_file_names[f]
            ))
            isotonic_model_dict_by_field[f] = bias_correction.read_file(
                isotonic_model_file_names[f]
            )

            imd = isotonic_model_dict_by_field[f]
            assert not imd[bias_correction.DO_UNCERTAINTY_CALIB_KEY]

    if uncertainty_calib_model_file_names is None:
        uncertainty_calib_model_dict_by_field = None
    else:
        uncertainty_calib_model_dict_by_field = [dict()] * num_fields

        for f in range(num_fields):
            print('Reading UC model from: "{0:s}"...'.format(
                uncertainty_calib_model_file_names[f]
            ))
            uncertainty_calib_model_dict_by_field[f] = (
                bias_correction.read_file(
                    uncertainty_calib_model_file_names[f]
                )
            )

            ucmd = uncertainty_calib_model_dict_by_field[f]
            assert ucmd[bias_correction.DO_UNCERTAINTY_CALIB_KEY]

    init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings
    ], dtype=int)

    input_file_names = prediction_io.find_files_for_period(
        directory_name=input_dir_name,
        first_init_time_unix_sec=init_time_limits_unix_sec[0],
        last_init_time_unix_sec=init_time_limits_unix_sec[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )
    print(SEPARATOR_STRING)

    for this_input_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_input_file_name))
        this_prediction_table_xarray = prediction_io.read_file(
            this_input_file_name
        )
        tptx = this_prediction_table_xarray

        if isotonic_model_file_names is not None:
            assert tptx.attrs[prediction_io.ISOTONIC_MODEL_FILES_KEY] is None
            assert (
                tptx.attrs[prediction_io.UNCERTAINTY_CALIB_MODEL_FILES_KEY]
                is None
            )

        if uncertainty_calib_model_file_names is not None:
            assert (
                tptx.attrs[prediction_io.UNCERTAINTY_CALIB_MODEL_FILES_KEY]
                is None
            )

            ucmd = uncertainty_calib_model_dict_by_field[0]
            if ucmd[bias_correction.DO_IR_BEFORE_UC_KEY]:
                assert (
                    isotonic_model_file_names is not None or
                    tptx.attrs[prediction_io.ISOTONIC_MODEL_FILES_KEY]
                    is not None
                )

        if isotonic_model_file_names is not None:
            this_prediction_table_xarray = bias_correction.apply_model_suite(
                prediction_table_xarray=this_prediction_table_xarray,
                model_dict_by_field=isotonic_model_dict_by_field,
                verbose=True
            )

        if uncertainty_calib_model_file_names is not None:
            this_prediction_table_xarray = bias_correction.apply_model_suite(
                prediction_table_xarray=this_prediction_table_xarray,
                model_dict_by_field=uncertainty_calib_model_dict_by_field,
                verbose=True
            )

        this_init_time_unix_sec = prediction_io.file_name_to_init_time(
            this_input_file_name
        )
        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=False
        )
        tptx = this_prediction_table_xarray

        print('Writing results to: "{0:s}"...'.format(this_output_file_name))
        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            target_matrix=tptx[prediction_io.TARGET_KEY].values,
            prediction_matrix=tptx[prediction_io.PREDICTION_KEY].values,
            latitude_matrix_deg_n=tptx[prediction_io.LATITUDE_KEY].values,
            longitude_matrix_deg_e=tptx[prediction_io.LONGITUDE_KEY].values,
            field_names=tptx[prediction_io.FIELD_NAME_KEY].values.tolist(),
            init_time_unix_sec=this_init_time_unix_sec,
            model_file_name=tptx.attrs[prediction_io.MODEL_FILE_KEY],
            isotonic_model_file_names=isotonic_model_file_names,
            uncertainty_calib_model_file_names=
            uncertainty_calib_model_file_names
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_TIME_LIMITS_ARG_NAME
        ),
        isotonic_model_file_names=getattr(
            INPUT_ARG_OBJECT, ISOTONIC_FILES_ARG_NAME
        ),
        uncertainty_calib_model_file_names=getattr(
            INPUT_ARG_OBJECT, UNCERTAINTY_CALIB_FILES_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
