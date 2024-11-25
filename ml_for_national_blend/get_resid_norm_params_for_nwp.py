"""Computes residual-normalization parameters for NWP data.

Residual-normalization parameters = stdev of temporal difference for each
variable.
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import nwp_model_io
import nwp_model_utils
import residual_normalization as resid_normalization
import get_norm_params_for_nwp as get_norm_params

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600

MODELS_ARG_NAME = 'model_names'
INPUT_DIRS_ARG_NAME = 'input_dir_name_by_model'
NON_RESID_NORM_FILE_ARG_NAME = 'input_non_resid_norm_file_name'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
INIT_TIME_COUNTS_ARG_NAME = 'num_init_times_by_model'
COMPUTE_INTERMEDIATE_ARG_NAME = 'compute_intermediate_params'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

MODELS_HELP_STRING = (
    'List of NWP models (each must be accepted by '
    '`nwp_model_utils.check_model_name`).'
)
INPUT_DIRS_HELP_STRING = (
    'List of directory paths, with the same length as {0:s}.  The [j]th item '
    'in this list should be the input path for the [j]th model, with files '
    'therein to be found by `interp_nwp_model_io.find_file` and read by '
    '`interp_nwp_model_io.read_file`.'
).format(
    MODELS_ARG_NAME
)
NON_RESID_NORM_FILE_HELP_STRING = (
    'Path to file with parameters for non-residual normalization.'
)
FIRST_TIME_HELP_STRING = (
    'First initialization time (format "yyyy-mm-dd-HH").  Normalization params '
    'will be based on all init times in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME
)
LAST_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_TIME_ARG_NAME
)
INIT_TIME_COUNTS_HELP_STRING = (
    'List with the same length as {0:s}.  The [j]th item in this list should '
    'be the number of init times for the [j]th model, to be randomly sampled '
    'from the period {1:s}...{2:s}.'
).format(
    MODELS_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME
)
COMPUTE_INTERMEDIATE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compute intermediate (final) normalization '
    'parameters.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`nwp_model_io.write_normalization_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODELS_ARG_NAME, type=str, nargs='+', required=True,
    help=MODELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NON_RESID_NORM_FILE_ARG_NAME, type=str, required=True,
    help=NON_RESID_NORM_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True,
    help=LAST_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_COUNTS_ARG_NAME, type=int, nargs='+', required=True,
    help=INIT_TIME_COUNTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPUTE_INTERMEDIATE_ARG_NAME, type=int, required=True,
    help=COMPUTE_INTERMEDIATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_names, input_dir_name_by_model, non_resid_norm_file_name,
         first_init_time_string, last_init_time_string, num_init_times_by_model,
         compute_intermediate_params, output_file_name):
    """Computes normalization parameters for NWP data.

    This is effectively the main method.

    :param model_names: See documentation at top of this script.
    :param input_dir_name_by_model: Same.
    :param non_resid_norm_file_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param num_init_times_by_model: Same.
    :param compute_intermediate_params: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    num_models = len(model_names)
    expected_dim = numpy.array([num_models], dtype=int)

    error_checking.assert_is_numpy_array(
        numpy.unique(numpy.array(model_names)),
        exact_dimensions=expected_dim
    )

    for this_model_name in model_names:
        nwp_model_utils.check_model_name(this_model_name)

    error_checking.assert_is_numpy_array(
        numpy.array(input_dir_name_by_model),
        exact_dimensions=expected_dim
    )

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )

    error_checking.assert_is_greater_numpy_array(num_init_times_by_model, 0)
    error_checking.assert_is_numpy_array(
        num_init_times_by_model, exact_dimensions=expected_dim
    )

    # Determine files to read for each model.
    interp_nwp_file_names = []

    for i in range(num_models):
        interp_nwp_file_names += get_norm_params._find_input_files_1model(
            model_name=model_names[i],
            first_init_time_unix_sec=first_init_time_unix_sec,
            last_init_time_unix_sec=last_init_time_unix_sec,
            input_dir_name=input_dir_name_by_model[i],
            num_init_times=num_init_times_by_model[i]
        )

    # Compute z-score parameters.
    if compute_intermediate_params:
        norm_param_table_xarray = (
            resid_normalization.get_intermediate_norm_params_for_nwp(
                interp_nwp_file_names=interp_nwp_file_names,
                non_resid_normalization_file_name=non_resid_norm_file_name,
                field_names=nwp_model_utils.ALL_FIELD_NAMES,
                precip_forecast_hours=
                get_norm_params._get_all_precip_forecast_hours(),
            )
        )
    else:
        norm_param_table_xarray = (
            resid_normalization.get_normalization_params_for_nwp(
                interp_nwp_file_names=interp_nwp_file_names,
                non_resid_normalization_file_name=non_resid_norm_file_name,
                field_names=nwp_model_utils.ALL_FIELD_NAMES,
                precip_forecast_hours=
                get_norm_params._get_all_precip_forecast_hours(),
            )
        )

    print('Writing z-score params to: "{0:s}"...'.format(output_file_name))
    nwp_model_io.write_normalization_file(
        norm_param_table_xarray=norm_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_names=getattr(INPUT_ARG_OBJECT, MODELS_ARG_NAME),
        input_dir_name_by_model=getattr(INPUT_ARG_OBJECT, INPUT_DIRS_ARG_NAME),
        non_resid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, NON_RESID_NORM_FILE_ARG_NAME
        ),
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_init_times_by_model=numpy.array(
            getattr(INPUT_ARG_OBJECT, INIT_TIME_COUNTS_ARG_NAME), dtype=int
        ),
        compute_intermediate_params=bool(
            getattr(INPUT_ARG_OBJECT, COMPUTE_INTERMEDIATE_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
