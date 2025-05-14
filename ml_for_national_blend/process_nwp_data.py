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
import sys
import copy
import shutil
import argparse
import warnings
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import nwp_model_io
import raw_nwp_model_io
import gfs_utils
import gefs_utils
import nwp_model_utils
import misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
MODEL_ARG_NAME = 'model_name'
TARGET_VARS_ONLY_ARG_NAME = 'target_vars_only'
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
TARGET_VARS_ONLY_HELP_STRING = (
    'Boolean flag.  If 1, will process only target variables.  If 0, will '
    'process all variables.'
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
    '--' + TARGET_VARS_ONLY_ARG_NAME, type=int, required=False, default=0,
    help=TARGET_VARS_ONLY_HELP_STRING
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


def _run(input_dir_name, model_name, target_vars_only,
         first_init_time_string, last_init_time_string,
         start_latitude_deg_n, end_latitude_deg_n,
         start_longitude_deg_e, end_longitude_deg_e,
         wgrib2_exe_name, temporary_dir_name, tar_output_files,
         output_dir_name):
    """Processes NWP data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param target_vars_only: Same.
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
    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
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
        forecast_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=model_name, init_time_unix_sec=this_init_time_unix_sec
        )
        num_forecast_hours = len(forecast_hours)

        # Find all input files (one GRIB2 file per forecast hour) for this
        # model run.
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

        # Write the output -- one xarray table for the whole model run -- to a
        # zarr file.
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

        # Put the zarr file inside a tar archive.  A "zarr file" actually
        # contains thousands of tiny files, which is hard on Hera (can easily
        # put you over the disk quota).  But if you store the whole thing in a
        # tar, Hera sees it as only one file, so you're in the clear!
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
        target_vars_only=bool(
            getattr(INPUT_ARG_OBJECT, TARGET_VARS_ONLY_ARG_NAME)
        ),
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
