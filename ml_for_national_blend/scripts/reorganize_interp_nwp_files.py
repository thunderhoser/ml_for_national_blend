"""Reorganizes interpolated NWP files into different directory structure."""

import re
import shutil
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_dir_name'
MODEL_ARG_NAME = 'model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Data in old directory structure will be found '
    'therein by `nwp_model_io.find_file`.'
)
MODEL_HELP_STRING = (
    'Name of NWP model (must be accepted by '
    '`nwp_model_utils.check_model_name`).'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First init time (format "yyyy-mm-dd-HH").  This script will process model '
    'runs initialized at all times in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Data in new directory structure will be '
    'written here (one NetCDF file per model run per lead time) by '
    '`interp_nwp_model_io.write_file`, to exact locations determined by '
    '`interp_nwp_model_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, model_name, first_init_time_string,
         last_init_time_string, output_dir_name):
    """Reorganizes interpolated NWP files into different directory structure.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
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
        allow_tar=True,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    for this_input_file_name in input_file_names:
        print('Reading data in old directory structure from: "{0:s}"...'.format(
            this_input_file_name
        ))
        nwp_forecast_table_xarray = nwp_model_io.read_file(
            zarr_file_name=this_input_file_name,
            allow_tar=True
        )

        nwpft = nwp_forecast_table_xarray
        forecast_hours = nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values

        for this_forecast_hour in forecast_hours:
            forecast_table_1hour_xarray = (
                nwp_model_utils.subset_by_forecast_hour(
                    nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                    desired_forecast_hours=
                    numpy.array([this_forecast_hour], dtype=int)
                )
            )

            output_file_name = interp_nwp_model_io.find_file(
                directory_name=output_dir_name,
                model_name=model_name,
                forecast_hour=this_forecast_hour,
                init_time_unix_sec=
                nwp_model_io.file_name_to_init_time(this_input_file_name),
                raise_error_if_missing=False
            )

            print((
                'Writing data in new directory structure to: "{0:s}"...'
            ).format(
                output_file_name
            ))
            interp_nwp_model_io.write_file(
                nwp_forecast_table_xarray=forecast_table_1hour_xarray,
                netcdf_file_name=output_file_name
            )

        if this_input_file_name.endswith('.tar'):
            this_input_file_name_zarr = re.sub(
                '.tar$', '.zarr', this_input_file_name
            )
            shutil.rmtree(this_input_file_name_zarr)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
