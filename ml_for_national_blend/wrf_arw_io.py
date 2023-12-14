"""Input/output methods for processed WRF-ARW model data.

Processed WRF-ARW data are stored with one zarr file per model run (init
time), containing forecasts valid at all lead times.
"""

import os
import sys
import shutil
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import file_system_utils
import error_checking
import wrf_arw_utils

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600


def find_file(directory_name, init_time_unix_sec, raise_error_if_missing=True):
    """Finds zarr file with one WRF-ARW model run (init time).

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: wrf_arw_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    wrf_arw_utils.check_init_time(init_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    wrf_arw_file_name = '{0:s}/wrf_arw_{1:s}.zarr'.format(
        directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT)
    )

    if raise_error_if_missing and not os.path.isdir(wrf_arw_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            wrf_arw_file_name
        )
        raise ValueError(error_string)

    return wrf_arw_file_name


def find_files_for_period(
        directory_name, first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many WRF-ARW files, each with one model run (init time).

    :param directory_name: Path to input directory.
    :param first_init_time_unix_sec: First initialization time in period.
    :param last_init_time_unix_sec: Last initialization time in period.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: zarr_file_names: 1-D list of paths to zarr files with WRF-ARW
        forecasts, one per model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=12 * HOURS_TO_SECONDS,
        include_endpoint=True
    )

    zarr_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_name = find_file(
            directory_name=directory_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isdir(this_file_name):
            zarr_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(zarr_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from init times {1:s} '
            'to {2:s}.'
        ).format(
            directory_name,
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[0], TIME_FORMAT
            ),
            time_conversion.unix_sec_to_string(
                init_times_unix_sec[-1], TIME_FORMAT
            )
        )
        raise ValueError(error_string)

    return zarr_file_names


def file_name_to_init_time(wrf_arw_file_name):
    """Parses initialization time from name of WRF-ARW file.

    :param wrf_arw_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    pathless_file_name = os.path.split(wrf_arw_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    init_time_string = extensionless_file_name.split('_')[-1]

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    wrf_arw_utils.check_init_time(init_time_unix_sec)

    return init_time_unix_sec


def read_file(zarr_file_name):
    """Reads WRF-ARW data from zarr file.

    :param zarr_file_name: Path to input file.
    :return: wrf_arw_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    return xarray.open_zarr(zarr_file_name)


def write_file(wrf_arw_table_xarray, zarr_file_name):
    """Writes WRF-ARW data to zarr file.

    :param wrf_arw_table_xarray: xarray table in format returned by `read_file`.
    :param zarr_file_name: Path to output file.
    """

    error_checking.assert_is_string(zarr_file_name)
    if os.path.isdir(zarr_file_name):
        shutil.rmtree(zarr_file_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=zarr_file_name
    )

    encoding_dict = {
        wrf_arw_utils.DATA_KEY: {'dtype': 'float32'}
    }
    wrf_arw_table_xarray.to_zarr(
        store=zarr_file_name, mode='w', encoding=encoding_dict
    )
