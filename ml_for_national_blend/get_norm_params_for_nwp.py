"""Computes normalization parameters for NWP data.

Normalization parameters = mean, stdev, and quantiles for each variable.
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
import time_periods
import error_checking
import nwp_model_io
import interp_nwp_model_io
import nwp_model_utils
import normalization

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600

MODELS_ARG_NAME = 'model_names'
INPUT_DIRS_ARG_NAME = 'input_dir_name_by_model'
FIRST_TIME_ARG_NAME = 'first_init_time_string'
LAST_TIME_ARG_NAME = 'last_init_time_string'
INIT_TIME_COUNTS_ARG_NAME = 'num_init_times_by_model'
NUM_QUANTILES_ARG_NAME = 'num_quantiles'
NUM_SAMPLE_VALUES_ARG_NAME = 'num_sample_values_per_file'
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
).format(MODELS_ARG_NAME)

FIRST_TIME_HELP_STRING = (
    'First initialization time (format "yyyy-mm-dd-HH").  Normalization params '
    'will be based on all init times in the continuous period {0:s}...{1:s}.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

LAST_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_TIME_ARG_NAME
)

INIT_TIME_COUNTS_HELP_STRING = (
    'List with the same length as {0:s}.  The [j]th item in this list should '
    'be the number of init times for the [j]th model, to be randomly sampled '
    'from the period {1:s}...{2:s}.'
).format(MODELS_ARG_NAME, FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_QUANTILES_HELP_STRING = (
    'Number of quantiles to store for each variable.  The quantile levels will '
    'be evenly spaced from 0 to 1 (i.e., the 0th to 100th percentile).'
)
NUM_SAMPLE_VALUES_HELP_STRING = (
    'Number of sample values per file to use for computing quantiles.  This '
    'value will be applied to each variable.'
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
    '--' + NUM_QUANTILES_ARG_NAME, type=int, required=False, default=1001,
    help=NUM_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLE_VALUES_ARG_NAME, type=int, required=True,
    help=NUM_SAMPLE_VALUES_ARG_NAME
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPUTE_INTERMEDIATE_ARG_NAME, type=int, required=True,
    help=COMPUTE_INTERMEDIATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _find_input_files_1model(
        model_name, first_init_time_unix_sec, last_init_time_unix_sec,
        input_dir_name, num_init_times):
    """Finds input files for one NWP model.

    :param model_name: Model name.
    :param first_init_time_unix_sec: First init time in period.
    :param last_init_time_unix_sec: Last init time in period.
    :param input_dir_name: Path to input directory for the given model.
    :param num_init_times: Number of init times to randomly subset from the
        period `first_init_time_unix_sec`...`last_init_time_unix_sec`.
    :return: interp_nwp_file_names: 1-D list of paths to input files for the
        given model.
    """

    init_time_interval_sec = nwp_model_utils.model_to_init_time_interval(
        model_name
    )
    init_time_interval_sec = max([init_time_interval_sec, 6 * HOURS_TO_SECONDS])

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=init_time_interval_sec,
        include_endpoint=True
    )

    # TODO(thunderhoser): This is a HACK.  I need to encode this RAP fuckery
    # somewhere else.
    if model_name == nwp_model_utils.RAP_MODEL_NAME:
        init_times_unix_sec += 3 * HOURS_TO_SECONDS

    good_indices = []

    for j in range(len(init_times_unix_sec)):
        print('Looking for {0:s} data at {1:s}...'.format(
            model_name.upper(),
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[j], TIME_FORMAT
            )
        ))

        forecast_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=model_name,
            init_time_unix_sec=init_times_unix_sec[j]
        )

        # TODO(thunderhoser): Checking only the first forecast hour is a HACK to
        # save time.
        this_file_name = interp_nwp_model_io.find_file(
            directory_name=input_dir_name,
            init_time_unix_sec=init_times_unix_sec[j],
            forecast_hour=forecast_hours[0],
            model_name=model_name,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_file_name):
            continue

        good_indices.append(j)

    init_times_unix_sec = [init_times_unix_sec[j] for j in good_indices]

    if len(init_times_unix_sec) > num_init_times:
        init_times_unix_sec = numpy.random.choice(
            init_times_unix_sec, size=num_init_times, replace=False
        )

    init_times_unix_sec = numpy.sort(init_times_unix_sec)
    interp_nwp_file_names = []

    for j in range(len(init_times_unix_sec)):
        print('Looking for {0:s} data at {1:s}...'.format(
            model_name.upper(),
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[j], TIME_FORMAT
            )
        ))

        forecast_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=model_name,
            init_time_unix_sec=init_times_unix_sec[j]
        )

        these_file_names = [
            interp_nwp_model_io.find_file(
                directory_name=input_dir_name,
                init_time_unix_sec=init_times_unix_sec[j],
                forecast_hour=h,
                model_name=model_name,
                raise_error_if_missing=False
            )
            for h in forecast_hours
        ]

        these_file_names = [f for f in these_file_names if os.path.isfile(f)]
        interp_nwp_file_names += these_file_names

    return interp_nwp_file_names


def _get_all_precip_forecast_hours():
    """Returns all possible forecast hours for precip.

    :return: precip_forecast_hours: 1-D numpy array.
    """

    dummy_init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=time_conversion.string_to_unix_sec(
            '2024-01-01-00', TIME_FORMAT
        ),
        end_time_unix_sec=time_conversion.string_to_unix_sec(
            '2024-01-01-23', TIME_FORMAT
        ),
        time_interval_sec=3600,
        include_endpoint=True
    )

    precip_forecast_hours = numpy.array([], dtype=int)

    for this_model_name in nwp_model_utils.ALL_MODEL_NAMES:
        for this_init_time_unix_sec in dummy_init_times_unix_sec:
            these_hours = nwp_model_utils.model_to_forecast_hours(
                model_name=this_model_name,
                init_time_unix_sec=this_init_time_unix_sec
            )
            precip_forecast_hours = numpy.concatenate([
                precip_forecast_hours, these_hours
            ])

    return numpy.unique(precip_forecast_hours)


def _run(model_names, input_dir_name_by_model,
         first_init_time_string, last_init_time_string,
         num_init_times_by_model, num_quantiles, num_sample_values_per_file,
         compute_intermediate_params, output_file_name):
    """Computes normalization parameters for NWP data.

    This is effectively the main method.

    :param model_names: See documentation at top of this script.
    :param input_dir_name_by_model: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param num_init_times_by_model: Same.
    :param num_quantiles: Same.
    :param num_sample_values_per_file: Same.
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
        interp_nwp_file_names += _find_input_files_1model(
            model_name=model_names[i],
            first_init_time_unix_sec=first_init_time_unix_sec,
            last_init_time_unix_sec=last_init_time_unix_sec,
            input_dir_name=input_dir_name_by_model[i],
            num_init_times=num_init_times_by_model[i]
        )

    # Compute z-score parameters.
    if compute_intermediate_params:
        norm_param_table_xarray = (
            normalization.get_intermediate_norm_params_for_nwp(
                interp_nwp_file_names=interp_nwp_file_names,
                field_names=nwp_model_utils.ALL_FIELD_NAMES,
                precip_forecast_hours=_get_all_precip_forecast_hours(),
                num_sample_values_per_file=num_sample_values_per_file
            )
        )
    else:
        norm_param_table_xarray = (
            normalization.get_normalization_params_for_nwp(
                interp_nwp_file_names=interp_nwp_file_names,
                field_names=nwp_model_utils.ALL_FIELD_NAMES,
                precip_forecast_hours=_get_all_precip_forecast_hours(),
                num_quantiles=num_quantiles,
                num_sample_values_per_file=num_sample_values_per_file
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
        first_init_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_init_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_init_times_by_model=numpy.array(
            getattr(INPUT_ARG_OBJECT, INIT_TIME_COUNTS_ARG_NAME), dtype=int
        ),
        num_quantiles=getattr(INPUT_ARG_OBJECT, NUM_QUANTILES_ARG_NAME),
        num_sample_values_per_file=getattr(
            INPUT_ARG_OBJECT, NUM_SAMPLE_VALUES_ARG_NAME
        ),
        compute_intermediate_params=bool(
            getattr(INPUT_ARG_OBJECT, COMPUTE_INTERMEDIATE_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
