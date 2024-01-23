"""Processes NWP data.

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Full domain
- Full resolution
- Variables: all keys in list `nwp_model_utils.ALL_FIELD_NAMES`

The output will contain the same data, in zarr format, with one file per model
run (init time).
"""

import os
import copy
import shutil
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import raw_nwp_model_io
from ml_for_national_blend.utils import gfs_utils
from ml_for_national_blend.utils import gefs_utils
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
MODEL_ARG_NAME = 'model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
START_LATITUDE_ARG_NAME = 'start_latitude_deg_n'
END_LATITUDE_ARG_NAME = 'end_latitude_deg_n'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
TAR_OUTPUTS_ARG_NAME = 'tar_output_files'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of main input directory, containing one GRIB2 file per model run '
    '(init time) and forecast hour (lead time).  Files therein will be found '
    'by `raw_nwp_model_io.find_file`.'
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

START_LATITUDE_HELP_STRING = (
    'Start latitude.  This script will process all latitudes in the '
    'contiguous domain {0:s}...{1:s}.'
).format(
    START_LATITUDE_ARG_NAME, END_LATITUDE_ARG_NAME
)
END_LATITUDE_HELP_STRING = 'Same as {0:s} but end latitude.'.format(
    START_LATITUDE_ARG_NAME
)
START_LONGITUDE_HELP_STRING = (
    'Start longitude.  This script will process all longitudes in the '
    'contiguous domain {0:s}...{1:s}.  This domain may cross the International '
    'Date Line.'
).format(
    START_LONGITUDE_ARG_NAME, END_LONGITUDE_ARG_NAME
)
END_LONGITUDE_HELP_STRING = 'Same as {0:s} but end longitude.'.format(
    START_LONGITUDE_ARG_NAME
)

WGRIB2_EXE_HELP_STRING = 'Path to wgrib2 executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib2.'
)
TAR_OUTPUTS_HELP_STRING = 'Boolean flag.  If 1, will tar output files.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'zarr file per model run) by `nwp_model_io.write_file`, to exact locations '
    'determined by `nwp_model_io.find_file`.'
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
    '--' + START_LATITUDE_ARG_NAME, type=float, required=False, default=1001,
    help=START_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_LATITUDE_ARG_NAME, type=float, required=False, default=1001,
    help=END_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + START_LONGITUDE_ARG_NAME, type=float, required=False, default=1001,
    help=START_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + END_LONGITUDE_ARG_NAME, type=float, required=False, default=1001,
    help=END_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WGRIB2_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB2_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TAR_OUTPUTS_ARG_NAME, type=int, required=False, default=0,
    help=TAR_OUTPUTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, model_name,
         first_init_time_string, last_init_time_string,
         start_latitude_deg_n, end_latitude_deg_n,
         start_longitude_deg_e, end_longitude_deg_e,
         wgrib2_exe_name, temporary_dir_name, tar_output_files,
         output_dir_name):
    """Processes NWP data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param start_latitude_deg_n: Same.
    :param end_latitude_deg_n: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param tar_output_files: Same.
    :param output_dir_name: Same.
    """

    if model_name not in [
            nwp_model_utils.GFS_MODEL_NAME, nwp_model_utils.GEFS_MODEL_NAME
    ]:
        start_latitude_deg_n = 1001.
        end_latitude_deg_n = 1001.
        start_longitude_deg_e = 1001.
        end_longitude_deg_e = 1001.

    these_coords = [
        start_latitude_deg_n, end_latitude_deg_n,
        start_longitude_deg_e, end_longitude_deg_e
    ]

    if any([c > 1000 for c in these_coords]):
        start_latitude_deg_n = None
        end_latitude_deg_n = None
        start_longitude_deg_e = None
        end_longitude_deg_e = None

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=nwp_model_utils.model_to_init_time_interval(
            model_name
        ),
        include_endpoint=True
    )

    latitude_matrix_deg_n = nwp_model_utils.read_model_coords(
        model_name=model_name
    )[0]
    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    if start_latitude_deg_n is None:
        desired_row_indices = numpy.linspace(
            0, num_grid_rows - 1, num=num_grid_rows, dtype=int
        )
        desired_column_indices = numpy.linspace(
            0, num_grid_columns - 1, num=num_grid_columns, dtype=int
        )
    elif model_name == nwp_model_utils.GFS_MODEL_NAME:
        desired_row_indices = gfs_utils.desired_latitudes_to_rows(
            start_latitude_deg_n=start_latitude_deg_n,
            end_latitude_deg_n=end_latitude_deg_n
        )
        desired_column_indices = gfs_utils.desired_longitudes_to_columns(
            start_longitude_deg_e=start_longitude_deg_e,
            end_longitude_deg_e=end_longitude_deg_e
        )
    else:
        desired_row_indices = gefs_utils.desired_latitudes_to_rows(
            start_latitude_deg_n=start_latitude_deg_n,
            end_latitude_deg_n=end_latitude_deg_n
        )
        desired_column_indices = gefs_utils.desired_longitudes_to_columns(
            start_longitude_deg_e=start_longitude_deg_e,
            end_longitude_deg_e=end_longitude_deg_e
        )

    read_incremental_precip = model_name in [
        nwp_model_utils.NAM_MODEL_NAME, nwp_model_utils.NAM_NEST_MODEL_NAME,
        nwp_model_utils.GEFS_MODEL_NAME
    ]

    field_names = copy.deepcopy(nwp_model_utils.ALL_FIELD_NAMES)
    field_names = set(field_names)
    if model_name == nwp_model_utils.GRIDDED_LAMP_MODEL_NAME:
        field_names = [
            nwp_model_utils.TEMPERATURE_2METRE_NAME,
            nwp_model_utils.DEWPOINT_2METRE_NAME,
            nwp_model_utils.WIND_GUST_10METRE_NAME,
            nwp_model_utils.U_WIND_10METRE_NAME,
            nwp_model_utils.V_WIND_10METRE_NAME
        ]
    else:
        field_names.remove(nwp_model_utils.WIND_GUST_10METRE_NAME)

    for this_init_time_unix_sec in init_times_unix_sec:
        forecast_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=model_name, init_time_unix_sec=this_init_time_unix_sec
        )
        num_forecast_hours = len(forecast_hours)

        # TODO(thunderhoser): For test scripts, shorten forecast-hour list here.
        if model_name in [nwp_model_utils.GEFS_MODEL_NAME, nwp_model_utils.GRIDDED_LAMP_MODEL_NAME]:
            forecast_hours = numpy.array([6, 24], dtype=int)

        input_file_names = [
            raw_nwp_model_io.find_file(
                directory_name=input_dir_name,
                model_name=model_name,
                init_time_unix_sec=this_init_time_unix_sec,
                forecast_hour=h,
                raise_error_if_missing=True
            )
            for h in forecast_hours
        ]

        nwp_forecast_tables_xarray = [None] * num_forecast_hours

        for k in range(num_forecast_hours):
            this_rotate_flag = model_name not in [
                nwp_model_utils.RAP_MODEL_NAME,
                nwp_model_utils.GFS_MODEL_NAME,
                nwp_model_utils.GEFS_MODEL_NAME,
                nwp_model_utils.GRIDDED_LAMP_MODEL_NAME
            ]

            nwp_forecast_tables_xarray[k] = raw_nwp_model_io.read_file(
                grib2_file_name=input_file_names[k],
                model_name=model_name,
                desired_row_indices=desired_row_indices,
                desired_column_indices=desired_column_indices,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name,
                field_names=field_names,
                rotate_winds=this_rotate_flag,
                read_incremental_precip=read_incremental_precip
            )

            print(SEPARATOR_STRING)

        nwp_forecast_table_xarray = nwp_model_utils.concat_over_forecast_hours(
            nwp_forecast_tables_xarray
        )
        if read_incremental_precip:
            nwp_forecast_table_xarray = (
                nwp_model_utils.precip_from_incremental_to_full_run(
                    nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                    model_name=model_name,
                    init_time_unix_sec=this_init_time_unix_sec
                )
            )

        nwp_forecast_table_xarray = nwp_model_utils.remove_negative_precip(
            nwp_forecast_table_xarray
        )

        if model_name == nwp_model_utils.RAP_MODEL_NAME:
            nwp_forecast_table_xarray = (
                nwp_model_utils.rotate_rap_winds_to_earth_relative(
                    nwp_forecast_table_xarray
                )
            )

        output_file_name = nwp_model_io.find_file(
            directory_name=output_dir_name,
            model_name=model_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        nwp_model_io.write_file(
            zarr_file_name=output_file_name,
            nwp_forecast_table_xarray=nwp_forecast_table_xarray
        )
        print(SEPARATOR_STRING)

        if not tar_output_files:
            continue

        output_file_name_tarred = '{0:s}.tar'.format(
            os.path.splitext(output_file_name)[0]
        )
        print('Creating tar file: "{0:s}"...'.format(output_file_name_tarred))

        misc_utils.create_tar_file(
            source_paths_to_tar=[output_file_name],
            tar_file_name=output_file_name_tarred
        )
        shutil.rmtree(output_file_name)


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
        start_latitude_deg_n=getattr(INPUT_ARG_OBJECT, START_LATITUDE_ARG_NAME),
        end_latitude_deg_n=getattr(INPUT_ARG_OBJECT, END_LATITUDE_ARG_NAME),
        start_longitude_deg_e=getattr(
            INPUT_ARG_OBJECT, START_LONGITUDE_ARG_NAME
        ),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        tar_output_files=bool(getattr(INPUT_ARG_OBJECT, TAR_OUTPUTS_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
