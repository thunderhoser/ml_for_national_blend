"""Input/output methods for NWP forecasts interpolated to NBM grid."""

import os
import sys
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import file_system_utils
import error_checking
import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'


def find_file(directory_name, init_time_unix_sec, forecast_hour, model_name,
              raise_error_if_missing=True):
    """Finds NetCDF file with NWP forecasts for one init time & one fcst hour.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param forecast_hour: Forecast hour.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: interp_nwp_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    interp_nwp_file_name = '{0:s}/{1:s}/{2:s}_{1:s}_hour{3:03d}.nc'.format(
        directory_name,
        time_conversion.unix_sec_to_string(init_time_unix_sec, TIME_FORMAT),
        model_name,
        forecast_hour
    )

    if os.path.isfile(interp_nwp_file_name) or not raise_error_if_missing:
        return interp_nwp_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        interp_nwp_file_name
    )
    raise ValueError(error_string)


def find_files_for_period(
        directory_name, model_name, forecast_hour,
        first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many NetCDF files, each with NWP forecasts for one run (init time).

    All files must pertain to the same model and forecast hour.

    :param directory_name: Path to input directory.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param forecast_hour: Forecast hour.
    :param first_init_time_unix_sec: First initialization time in period.
    :param last_init_time_unix_sec: Last initialization time in period.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: interp_nwp_file_names: 1-D list of paths to NetCDF files with NWP
        forecasts, one per model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=
        nwp_model_utils.model_to_init_time_interval(model_name),
        include_endpoint=True
    )

    interp_nwp_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_name = find_file(
            directory_name=directory_name,
            init_time_unix_sec=this_init_time_unix_sec,
            forecast_hour=forecast_hour,
            model_name=model_name,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isfile(this_file_name):
            interp_nwp_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(interp_nwp_file_names) == 0:
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

    return interp_nwp_file_names


def file_name_to_init_time(interp_nwp_file_name):
    """Parses initialization time from name of file with NWP forecasts.

    :param interp_nwp_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    pathless_file_name = os.path.split(interp_nwp_file_name)[1]
    init_time_string = pathless_file_name.split('_')[2]
    model_name = '_'.join(pathless_file_name.split('_')[:-2])

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT
    )
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )

    return init_time_unix_sec


def file_name_to_model_name(interp_nwp_file_name):
    """Parses model name from name of file with NWP forecasts.

    :param interp_nwp_file_name: File path.
    :return: model_name: Model name.
    """

    pathless_file_name = os.path.split(interp_nwp_file_name)[1]
    model_name = '_'.join(pathless_file_name.split('_')[:-2])

    nwp_model_utils.check_model_name(model_name)
    return model_name


def file_name_to_forecast_hour(interp_nwp_file_name):
    """Parses forecast hour from name of file with NWP forecasts.

    :param interp_nwp_file_name: File path.
    :return: forecast_hour: Forecast hour.
    """

    pathless_file_name = os.path.split(interp_nwp_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    forecast_hour = int(
        extensionless_file_name.split('_')[-1].replace('hour', '')
    )

    error_checking.assert_is_greater(forecast_hour, 0)
    return forecast_hour


def read_file(netcdf_file_name):
    """Reads interpolated NWP forecasts from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: nwp_forecast_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)


def write_file(nwp_forecast_table_xarray, netcdf_file_name):
    """Writes interpolated NWP forecasts to NetCDF file.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    encoding_dict = {
        nwp_model_utils.DATA_KEY: {'dtype': 'float32'}
    }
    nwp_forecast_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF4_CLASSIC',
        encoding=encoding_dict
    )