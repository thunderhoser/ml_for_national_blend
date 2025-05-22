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
import argparse
import warnings
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import time_periods
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import raw_nwp_model_io
from ml_for_national_blend.utils import gfs_utils
from ml_for_national_blend.utils import gefs_utils
from ml_for_national_blend.utils import nwp_model_utils

NONE_STRINGS = ['', 'none', 'None']
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
INPUT_FILE_ARG_NAME = 'input_grib2_file_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
MODEL_ARG_NAME = 'model_name'
TARGET_VARS_ONLY_ARG_NAME = 'target_vars_only'
START_LATITUDE_ARG_NAME = 'start_latitude_deg_n'
END_LATITUDE_ARG_NAME = 'end_latitude_deg_n'
START_LONGITUDE_ARG_NAME = 'start_longitude_deg_e'
END_LONGITUDE_ARG_NAME = 'end_longitude_deg_e'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing one GRIB2 file per model run '
    '(init time) and forecast hour (lead time).  Files therein will be found '
    'by `raw_nwp_model_io.find_file`.  If you would rather specify a single '
    'GRIB file, leave this argument alone and use `{0:s}`.'
).format(
    INPUT_FILE_ARG_NAME
)
INPUT_FILE_HELP_STRING = (
    'Path to single input file, containing data for one model run (init time) '
    'and one forecast hour (lead time).  If you would rather work on many '
    'files, leave this argument alone and use `{0:s}`.'
).format(
    INPUT_DIR_ARG_NAME
)
FIRST_INIT_TIME_HELP_STRING = (
    '[used only if `{0:s}` is specified, not if `{1:s}` is specified] '
    'First init time (format "yyyy-mm-dd-HH").  This script will process model '
    'runs initialized at all times in the continuous period {2:s}...{3:s}.'
).format(
    INPUT_DIR_ARG_NAME, INPUT_FILE_ARG_NAME,
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
MODEL_HELP_STRING = (
    'Name of NWP model (must be accepted by '
    '`nwp_model_utils.check_model_name`).'
)
TARGET_VARS_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, will process only target variables.  If 0, will '
    'process all variables.'
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
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'NetCDF file per init time and forecast hour) by '
    '`nwp_model_io.write_file`, to exact locations determined by '
    '`nwp_model_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=False, default='',
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=False, default='',
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=False, default='',
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=False, default='',
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_VARS_ONLY_ARG_NAME, type=int, required=False, default=0,
    help=TARGET_VARS_ONLY_HELP_STRING
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, input_file_name,
         first_init_time_string, last_init_time_string,
         model_name, target_vars_only,
         start_latitude_deg_n, end_latitude_deg_n,
         start_longitude_deg_e, end_longitude_deg_e,
         wgrib2_exe_name, temporary_dir_name, output_dir_name):
    """Processes NWP data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param input_file_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param model_name: Same.
    :param target_vars_only: Same.
    :param start_latitude_deg_n: Same.
    :param end_latitude_deg_n: Same.
    :param start_longitude_deg_e: Same.
    :param end_longitude_deg_e: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_dir_name: Same.
    """

    if input_dir_name in NONE_STRINGS:
        input_dir_name = None
    else:
        input_file_name = None

    if input_file_name in NONE_STRINGS:
        input_file_name = None
    else:
        input_dir_name = None

    assert not (input_dir_name is None and input_file_name is None)

    # start_latitude_deg_n, end_latitude_deg_n, start_longitude_deg_e, and
    # end_longitude_deg_e are used to subset the domain for global models.
    # The only global models are GFS, GEFS, and ECMWF.  Thus, if the model is
    # anything else, subsetting the domain is not needed.
    if model_name not in [
            nwp_model_utils.GFS_MODEL_NAME, nwp_model_utils.GEFS_MODEL_NAME,
            nwp_model_utils.ECMWF_MODEL_NAME
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

    # For the given model, find all initialization times in the specified period
    # (first_init_time_unix_sec to last_init_time_unix_sec).
    if input_file_name is None:
        first_init_time_unix_sec = time_conversion.string_to_unix_sec(
            first_init_time_string, TIME_FORMAT
        )
        last_init_time_unix_sec = time_conversion.string_to_unix_sec(
            last_init_time_string, TIME_FORMAT
        )
    else:
        first_init_time_unix_sec = raw_nwp_model_io.file_name_to_init_time(
            nwp_forecast_file_name=input_file_name,
            model_name=model_name
        )
        last_init_time_unix_sec = first_init_time_unix_sec + 0

    init_time_interval_sec = nwp_model_utils.model_to_init_time_interval(
        model_name
    )

    # TODO(thunderhoser): This is a HACK, because currently I don't want to
    # process all the data.
    init_time_interval_sec = numpy.maximum(
        init_time_interval_sec, 6 * HOURS_TO_SECONDS
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=init_time_interval_sec,
        include_endpoint=True
    )

    # TODO(thunderhoser): This is a HACK, because I want to use off-synoptic
    # times for the RAP.
    if model_name == nwp_model_utils.RAP_MODEL_NAME:
        init_times_unix_sec += 3 * HOURS_TO_SECONDS

    # For the given model, read grid coordinates and find desired grid points.
    # If the model is global (GFS/GEFS/ECMWF), desired grid points may be a
    # subset.  Otherwise, desired grid points will be the entire grid.
    latitude_matrix_deg_n = nwp_model_utils.read_model_coords(
        model_name=model_name
    )[0]
    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    if start_latitude_deg_n is None:  # Desired grid points are entire grid.
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
    elif model_name == nwp_model_utils.GEFS_MODEL_NAME:
        desired_row_indices = gefs_utils.desired_latitudes_to_rows(
            start_latitude_deg_n=start_latitude_deg_n,
            end_latitude_deg_n=end_latitude_deg_n
        )
        desired_column_indices = gefs_utils.desired_longitudes_to_columns(
            start_longitude_deg_e=start_longitude_deg_e,
            end_longitude_deg_e=end_longitude_deg_e
        )
    elif model_name == nwp_model_utils.ECMWF_MODEL_NAME:
        desired_row_indices = gfs_utils.desired_latitudes_to_rows(
            start_latitude_deg_n=start_latitude_deg_n,
            end_latitude_deg_n=end_latitude_deg_n
        )
        desired_column_indices = gfs_utils.desired_longitudes_to_columns(
            start_longitude_deg_e=start_longitude_deg_e,
            end_longitude_deg_e=end_longitude_deg_e
        )

    # Three models -- the NAM, NAM Nest, and GEFS -- store incremental precip.
    # This means that, at forecast hour H, the precip variable (APCP) is an
    # accumulation between H and some previous forecast hour.
    #
    # Other models store full-run precip, which means that at forecast hour H,
    # the precip variable is an accumulation between forecast hours 0 and H --
    # i.e., over the full model run.
    read_incremental_precip = model_name in [
        nwp_model_utils.NAM_MODEL_NAME,
        nwp_model_utils.NAM_NEST_MODEL_NAME,
        nwp_model_utils.GEFS_MODEL_NAME
    ]

    # Gridded LAMP has only 5 output variables; the other models have a bunch.
    if model_name == nwp_model_utils.GRIDDED_LAMP_MODEL_NAME:
        field_names = [
            nwp_model_utils.TEMPERATURE_2METRE_NAME,
            nwp_model_utils.DEWPOINT_2METRE_NAME,
            nwp_model_utils.WIND_GUST_10METRE_NAME,
            nwp_model_utils.U_WIND_10METRE_NAME,
            nwp_model_utils.V_WIND_10METRE_NAME
        ]
    elif model_name == nwp_model_utils.GRIDDED_MOS_MODEL_NAME:
        field_names = [
            nwp_model_utils.TEMPERATURE_2METRE_NAME,
            nwp_model_utils.DEWPOINT_2METRE_NAME,
            nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME,
            nwp_model_utils.WIND_GUST_10METRE_NAME,
            nwp_model_utils.U_WIND_10METRE_NAME,
            nwp_model_utils.V_WIND_10METRE_NAME
        ]

        if target_vars_only:
            field_names = set(field_names)
            field_names.remove(nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME)
            field_names = list(field_names)
    else:
        if target_vars_only:
            field_names = [
                nwp_model_utils.TEMPERATURE_2METRE_NAME,
                nwp_model_utils.DEWPOINT_2METRE_NAME,
                nwp_model_utils.WIND_GUST_10METRE_NAME,
                nwp_model_utils.U_WIND_10METRE_NAME,
                nwp_model_utils.V_WIND_10METRE_NAME
            ]
        else:
            field_names = copy.deepcopy(nwp_model_utils.ALL_FIELD_NAMES)

        field_names = set(field_names)
        field_names.remove(nwp_model_utils.WIND_GUST_10METRE_NAME)
        field_names = list(field_names)

    # For each initialization time (i.e., for each model run)...
    for this_init_time_unix_sec in init_times_unix_sec:

        # For the given model and init time, determine which forecast hours
        # should be available.
        if input_file_name is None:
            forecast_hours = nwp_model_utils.model_to_forecast_hours(
                model_name=model_name,
                init_time_unix_sec=this_init_time_unix_sec
            )
        else:
            forecast_hour = raw_nwp_model_io.file_name_to_forecast_hour(
                input_file_name
            )
            forecast_hours = numpy.array([forecast_hour], dtype=int)

        num_forecast_hours = len(forecast_hours)

        # Find all input files (one GRIB2 file per forecast hour) for this
        # model run.
        if input_file_name is None:
            input_file_names = [
                raw_nwp_model_io.find_file(
                    directory_name=input_dir_name,
                    model_name=model_name,
                    init_time_unix_sec=this_init_time_unix_sec,
                    forecast_hour=h,
                    raise_error_if_missing=False
                )
                for h in forecast_hours
            ]
        else:
            input_file_names = [input_file_name]

        found_all_inputs = all([os.path.isfile(f) for f in input_file_names])
        continue_flag = False
        be_lenient_with_forecast_hours = False
        using_old_gfs_or_gefs = False
        using_oldish_gfs = False

        if not found_all_inputs:
            continue_flag = True

            bad_file_names = [
                f for f in input_file_names if not os.path.isfile(f)
            ]
            warning_string = (
                'POTENTIAL ERROR: Could not find all input files for the given '
                'init time.  The following files are missing:\n{0:s}'
            ).format(str(bad_file_names))

            warnings.warn(warning_string)

            if model_name == nwp_model_utils.HRRR_MODEL_NAME:
                short_range_indices = numpy.where(forecast_hours <= 18)[0]
                found_all_short_range_inputs = all([
                    os.path.isfile(input_file_names[k])
                    for k in short_range_indices
                ])

                if found_all_short_range_inputs:
                    continue_flag = False
                    be_lenient_with_forecast_hours = True

            if model_name in [
                    nwp_model_utils.NAM_NEST_MODEL_NAME,
                    nwp_model_utils.WRF_ARW_MODEL_NAME
            ]:
                short_range_indices = numpy.where(forecast_hours <= 24)[0]
                found_all_short_range_inputs = all([
                    os.path.isfile(input_file_names[k])
                    for k in short_range_indices
                ])

                if found_all_short_range_inputs:
                    continue_flag = False
                    be_lenient_with_forecast_hours = True

            if model_name == nwp_model_utils.RAP_MODEL_NAME:
                short_range_indices = numpy.where(forecast_hours <= 21)[0]
                found_all_short_range_inputs = all([
                    os.path.isfile(input_file_names[k])
                    for k in short_range_indices
                ])

                if found_all_short_range_inputs:
                    continue_flag = False
                    be_lenient_with_forecast_hours = True

            if model_name == nwp_model_utils.GFS_MODEL_NAME:
                oldish_forecast_hours = (
                    nwp_model_utils.model_to_oldish_forecast_hours(model_name)
                )
                essential_indices = numpy.where(numpy.isin(
                    element=forecast_hours, test_elements=oldish_forecast_hours
                ))[0]
                found_all_essential_inputs = all([
                    os.path.isfile(input_file_names[k])
                    for k in essential_indices
                ])

                if found_all_essential_inputs:
                    continue_flag = False
                    be_lenient_with_forecast_hours = True
                    using_oldish_gfs = True

            if continue_flag and model_name in [
                    nwp_model_utils.GFS_MODEL_NAME,
                    nwp_model_utils.GEFS_MODEL_NAME
            ]:
                old_forecast_hours = (
                    nwp_model_utils.model_to_old_forecast_hours(model_name)
                )
                essential_indices = numpy.where(numpy.isin(
                    element=forecast_hours, test_elements=old_forecast_hours
                ))[0]
                found_all_essential_inputs = all([
                    os.path.isfile(input_file_names[k])
                    for k in essential_indices
                ])

                if found_all_essential_inputs:
                    continue_flag = False
                    be_lenient_with_forecast_hours = True
                    read_incremental_precip = True
                    using_old_gfs_or_gefs = True

        if continue_flag:
            continue
        if target_vars_only:
            read_incremental_precip = False
        if input_dir_name is None:
            read_incremental_precip = False

        nwp_forecast_tables_xarray = [None] * num_forecast_hours

        for k in range(num_forecast_hours):
            if not os.path.isfile(input_file_names[k]):
                continue

            # Most models store wind vectors in grid-relative coordinates.
            # These vectors must be rotated to Earth-relative coordinates.
            this_rotate_flag = model_name not in [
                nwp_model_utils.RAP_MODEL_NAME,
                nwp_model_utils.GFS_MODEL_NAME,
                nwp_model_utils.GEFS_MODEL_NAME,
                nwp_model_utils.ECMWF_MODEL_NAME,
                nwp_model_utils.GRIDDED_LAMP_MODEL_NAME
            ]

            # raw_nwp_model_io.read_file does all the dirty work.
            # It reads all the desired fields, converts them to SI units
            # (if necessary), rotates wind vectors to Earth-relative
            # (if necessary), and subsets the global grid (if necessary).
            if using_old_gfs_or_gefs:
                nwp_forecast_tables_xarray[k] = (
                    raw_nwp_model_io.read_old_gfs_or_gefs_file(
                        grib2_file_name=input_file_names[k],
                        model_name=model_name,
                        desired_row_indices=desired_row_indices,
                        desired_column_indices=desired_column_indices,
                        wgrib2_exe_name=wgrib2_exe_name,
                        temporary_dir_name=temporary_dir_name,
                        field_names=field_names
                    )
                )
            elif using_oldish_gfs:
                nwp_forecast_tables_xarray[k] = (
                    raw_nwp_model_io.read_oldish_gfs_file(
                        grib2_file_name=input_file_names[k],
                        model_name=model_name,
                        desired_row_indices=desired_row_indices,
                        desired_column_indices=desired_column_indices,
                        wgrib2_exe_name=wgrib2_exe_name,
                        temporary_dir_name=temporary_dir_name,
                        field_names=field_names,
                    )
                )
            elif model_name == nwp_model_utils.ECMWF_MODEL_NAME:
                nwp_forecast_tables_xarray[k] = (
                    raw_nwp_model_io.read_ecmwf_file(
                        grib_file_name=input_file_names[k],
                        desired_row_indices=desired_row_indices,
                        desired_column_indices=desired_column_indices,
                        wgrib_exe_name=wgrib2_exe_name,
                        temporary_dir_name=temporary_dir_name,
                        field_names=field_names
                    )
                )
            else:
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

        # The above for-loop creates one xarray table per forecast hour.
        # Concatenate these all into one table.
        nwp_forecast_tables_xarray = [
            t for t in nwp_forecast_tables_xarray if t is not None
        ]
        nwp_forecast_table_xarray = nwp_model_utils.concat_over_forecast_hours(
            nwp_forecast_tables_xarray
        )

        # If necessary, convert incremental precip to full-model-run precip.
        if read_incremental_precip:
            if using_old_gfs_or_gefs:
                nwp_forecast_table_xarray = (
                    nwp_model_utils.old_gfs_or_gefs_precip_from_incr_to_full(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        model_name=model_name
                    )
                )
            else:
                nwp_forecast_table_xarray = (
                    nwp_model_utils.precip_from_incremental_to_full_run(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        model_name=model_name,
                        init_time_unix_sec=this_init_time_unix_sec,
                        be_lenient_with_forecast_hours=
                        be_lenient_with_forecast_hours
                    )
                )

        # This shouldn't happen, but I've experienced it in the past with GFS
        # data in GRIB2 format.
        nwp_forecast_table_xarray = nwp_model_utils.remove_negative_precip(
            nwp_forecast_table_xarray
        )

        # In general, if the model stores wind in grid-relative coordinates,
        # I just pass `rotate_winds=True` to raw_nwp_model_io.read_file.
        # Inside raw_nwp_model_io.read_file, I call wgrib2 to rotate all the
        # wind vectors inside the file.
        # However, this does not work for the RAP model, because the RAP GRIB2
        # files do not contain proper metadata about the grid.  Thus, I have to
        # use my own method for the RAP winds.
        if model_name == nwp_model_utils.RAP_MODEL_NAME:
            nwp_forecast_table_xarray = (
                nwp_model_utils.rotate_rap_winds_to_earth_relative(
                    nwp_forecast_table_xarray
                )
            )

        # Write the output to NetCDF files -- one file per forecast hour.
        remaining_forecast_hours = nwp_forecast_table_xarray.coords[
            nwp_model_utils.FORECAST_HOUR_DIM
        ].values

        for this_forecast_hour in remaining_forecast_hours:
            output_file_name = nwp_model_io.find_file(
                directory_name=output_dir_name,
                init_time_unix_sec=this_init_time_unix_sec,
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

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        target_vars_only=bool(
            getattr(INPUT_ARG_OBJECT, TARGET_VARS_ONLY_ARG_NAME)
        ),
        start_latitude_deg_n=getattr(INPUT_ARG_OBJECT, START_LATITUDE_ARG_NAME),
        end_latitude_deg_n=getattr(INPUT_ARG_OBJECT, END_LATITUDE_ARG_NAME),
        start_longitude_deg_e=getattr(
            INPUT_ARG_OBJECT, START_LONGITUDE_ARG_NAME
        ),
        end_longitude_deg_e=getattr(INPUT_ARG_OBJECT, END_LONGITUDE_ARG_NAME),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
