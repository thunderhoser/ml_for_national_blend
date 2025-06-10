"""Converts NWP-forecast precipitation from incremental to full-run values.

This is a really stupid process, necessitated by the fact that raw NWP output
(in GRIB2 files) contains incremental values (from one forecast hour to the
next) instead of full-run values.  Incremental values would be okay, *if* the
time interval for incremental values were consistent across models.  For
example, if the raw GRIB2 files for every model contained 6-hour accumulations,
there would be no need to convert, because the accumulation period would be
consistent across models.  But instead, some models contain 1-hour
accumulations; some contain 3-hour accumulations; some contain 6-hour
accumulations; and some models even have DIFFERENT accumulations periods as the
forecast hour changes.  For example, some models might have 1-hour accumulations
early in the forecast horizon, increasing to 6-hour accumulations late in the
forecast horizon.  Yay for data standards in meteorology!!

This script -- which "knows" the accumulation period for every pair of model and
forecast hour -- converts the awful, inconsistent incremental precip values to
full-run precip values (accumulated since forecast hour 0 of the model run).
This allows precip values to be compared across models.

Note that this script should be run immediately after process_nwp_data.py -- if
and only if process_nwp_data.py was run on a single GRIB2 file rather than a
whole directory of GRIB2 files.  If process_nwp_data.py was run on a whole
directory of GRIB2 files (containing all forecast hours), then precip has
already been converted and there is nothing to do.
"""

import os
import argparse
import numpy
import xarray
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_netcdf_dir_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
MODEL_ARG_NAME = 'model_name'
OUTPUT_DIR_ARG_NAME = 'output_netcdf_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing NetCDF files produced by '
    'process_nwp_data.py.  Files within this directory will be found by '
    '`nwp_model_io.find_file` and read by `nwp_model_io.read_file`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First initialization time in period (format yyyy-mm-dd-HH).  This script '
    'will convert incremental to full-run precip for all initialization times '
    '(for the given model) in the continuous period `{0:s}`...`{1:s}`.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
MODEL_HELP_STRING = (
    'Name of model (must be accepted by `nwp_model_utils.check_model_name`).'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  NetCDF files (in the same format, but '
    'containing full-run instead of incremental precip) will be written here '
    'by `nwp_model_io.write_file`, to exact locations determined by '
    '`nwp_model_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
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
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _convert_one_model_run(input_dir_name, init_time_unix_sec, model_name,
                           output_dir_name):
    """Does the conversion for a single model run (init time).

    :param input_dir_name: See documentation at top of this file.
    :param init_time_unix_sec: Initialization time for the single model run.
    :param model_name: See documentation at top of this file.
    :param output_dir_name: Same.
    """

    forecast_hours = nwp_model_utils.model_to_forecast_hours(
        model_name=model_name, init_time_unix_sec=init_time_unix_sec
    )

    num_forecast_hours = len(forecast_hours)
    input_file_names = [''] * num_forecast_hours

    for j in range(num_forecast_hours):
        input_file_names[j] = nwp_model_io.find_file(
            directory_name=input_dir_name,
            init_time_unix_sec=init_time_unix_sec,
            forecast_hour=forecast_hours[j],
            model_name=model_name,
            raise_error_if_missing=False
        )

        if os.path.isfile(input_file_names[j]):
            continue

        input_file_names = input_file_names[:j]
        break

    if len(input_file_names) == 0:
        error_string = (
            'Could not find any forecast hours for model {0:s} init {1:s} in '
            'directory: "{2:s}"'
        ).format(
            model_name,
            time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT),
            input_dir_name
        )

        raise ValueError(error_string)

    num_forecast_hours = len(input_file_names)
    del forecast_hours
    nwp_forecast_tables_xarray = [xarray.Dataset()] * num_forecast_hours

    for j in range(num_forecast_hours):
        print('Reading data from: "{0:s}"...'.format(input_file_names[j]))
        nwp_forecast_tables_xarray[j] = nwp_model_io.read_file(
            input_file_names[j]
        )

    nwp_forecast_table_xarray = nwp_model_utils.concat_over_forecast_hours(
        nwp_forecast_tables_xarray
    )
    del nwp_forecast_tables_xarray

    # Ensure that precip hasn't already been converted from incremental to
    # full-run.  precip_matrix has dimensions fcst_hour x row x column.
    k = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.PRECIP_NAME
    )[0][0]

    precip_matrix = (
        nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values[..., k]
    )
    precip_time_diff_matrix = numpy.diff(precip_matrix, axis=0)

    if numpy.all(precip_time_diff_matrix > -1 * TOLERANCE):
        error_string = (
            'All precip differences (between one time step and the next) are '
            'non-negative, which suggests that precip has ALREADY been '
            'converted from incremental to full-run.  It is highly unlikely '
            'that precip values are still incremental, so doing this '
            '"conversion" again would lead to erroneous values.'
        )

        raise ValueError(error_string)

    # Do the conversion.
    nwp_forecast_table_xarray = (
        nwp_model_utils.real_time_precip_from_incr_to_full(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            model_name=model_name,
            init_time_unix_sec=init_time_unix_sec,
            be_lenient_with_forecast_hours=True
        )
    )
    nwp_forecast_table_xarray = nwp_model_utils.remove_negative_precip(
        nwp_forecast_table_xarray
    )

    forecast_hours = nwp_forecast_table_xarray.coords[
        nwp_model_utils.FORECAST_HOUR_DIM
    ].values

    for this_forecast_hour in forecast_hours:
        output_file_name = nwp_model_io.find_file(
            directory_name=output_dir_name,
            init_time_unix_sec=init_time_unix_sec,
            forecast_hour=this_forecast_hour,
            model_name=model_name,
            raise_error_if_missing=False
        )

        output_table_xarray = nwp_forecast_table_xarray.sel({
            nwp_model_utils.FORECAST_HOUR_DIM:
                numpy.array([this_forecast_hour], dtype=int)
        })

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        nwp_model_io.write_file(
            netcdf_file_name=output_file_name,
            nwp_forecast_table_xarray=output_table_xarray
        )


def _run(input_dir_name, first_init_time_string, last_init_time_string,
         model_name, output_dir_name):
    """Converts NWP-forecast precipitation from incremental to full-run values.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param model_name: Same.
    :param output_dir_name: Same.
    """

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )

    input_file_names = nwp_model_io.find_files_for_period(
        directory_name=input_dir_name,
        model_name=model_name,
        first_init_time_unix_sec=first_init_time_unix_sec,
        last_init_time_unix_sec=last_init_time_unix_sec,
        raise_error_if_all_missing=True,
        raise_error_if_any_missing=False
    )
    input_file_names.sort()

    init_times_unix_sec = numpy.array(
        [nwp_model_io.file_name_to_init_time(f) for f in input_file_names],
        dtype=int
    )
    init_times_unix_sec = numpy.unique(init_times_unix_sec)

    for this_init_time_unix_sec in init_times_unix_sec:
        _convert_one_model_run(
            input_dir_name=input_dir_name,
            init_time_unix_sec=this_init_time_unix_sec,
            model_name=model_name,
            output_dir_name=output_dir_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
