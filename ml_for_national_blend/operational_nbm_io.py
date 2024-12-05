"""Input/output methods for processed forecast data from operational NBM.

Each processed file should be a NetCDF file with the following properties:

- One model run (init time)
- One forecast hour (valid time)
- Full domain
- Full resolution
- Variables: all keys in dict `raw_operational_nbm_io.FIELD_NAME_TO_GRIB_NAME`
"""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import interp_nwp_model_io
import number_rounding
import time_conversion
import time_periods
import error_checking

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600


def find_file(directory_name, init_time_unix_sec, forecast_hour,
              raise_error_if_missing=True):
    """Finds NetCDF file with oper'l NBM fcsts for 1 init time & 1 valid time.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param forecast_hour: Forecast hour.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: operational_nbm_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    rounded_init_time_unix_sec = number_rounding.round_to_nearest(
        init_time_unix_sec, HOURS_TO_SECONDS
    )
    rounded_init_time_unix_sec = rounded_init_time_unix_sec.astype(int)
    error_checking.assert_equals(init_time_unix_sec, rounded_init_time_unix_sec)

    operational_nbm_file_name = (
        '{0:s}/{1:s}/operational_nbm_{1:s}_hour{2:03d}.nc'
    ).format(
        directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT),
        forecast_hour
    )

    if os.path.isfile(operational_nbm_file_name) or not raise_error_if_missing:
        return operational_nbm_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        operational_nbm_file_name
    )
    raise ValueError(error_string)


def find_files_for_period(
        directory_name, forecast_hour,
        first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many NetCDF files, each with op-NBM fcsts for one run (init time).

    All files must pertain to the same forecast hour.

    :param directory_name: Path to input directory.
    :param forecast_hour: Forecast hour.
    :param first_init_time_unix_sec: First initialization time in period.
    :param last_init_time_unix_sec: Last initialization time in period.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: operational_nbm_file_names: 1-D list of paths to NetCDF files with NWP
        forecasts, one per model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=HOURS_TO_SECONDS,
        include_endpoint=True
    )

    operational_nbm_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_name = find_file(
            directory_name=directory_name,
            init_time_unix_sec=this_init_time_unix_sec,
            forecast_hour=forecast_hour,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isfile(this_file_name):
            operational_nbm_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(operational_nbm_file_names) == 0:
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

    return operational_nbm_file_names


def file_name_to_init_time(operational_nbm_file_name):
    """Parses initialization time from name of file with op-NBM forecasts.

    :param operational_nbm_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    pathless_file_name = os.path.split(operational_nbm_file_name)[1]
    init_time_string = pathless_file_name.split('_')[-2]
    return time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )


def file_name_to_forecast_hour(operational_nbm_file_name):
    """Parses forecast hour from name of file with op-NBM forecasts.

    :param operational_nbm_file_name: File path.
    :return: forecast_hour: Forecast hour.
    """

    pathless_file_name = os.path.split(operational_nbm_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    forecast_hour = int(
        extensionless_file_name.split('_')[-1].replace('hour', '')
    )

    error_checking.assert_is_greater(forecast_hour, 0)
    return forecast_hour


def read_file(netcdf_file_name):
    """Reads operational NBM forecasts from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: op_nbm_forecast_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return interp_nwp_model_io.read_file(netcdf_file_name)


def write_file(op_nbm_forecast_table_xarray, netcdf_file_name):
    """Writes operational NBM forecasts to NetCDF file.

    :param op_nbm_forecast_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    interp_nwp_model_io.write_file(
        nwp_forecast_table_xarray=op_nbm_forecast_table_xarray,
        netcdf_file_name=netcdf_file_name
    )
