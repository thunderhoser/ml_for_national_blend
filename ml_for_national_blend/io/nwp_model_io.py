"""Input/output methods for processed NWP-model data."""

import glob
import xarray
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import time_periods
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

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
    :return: nwp_forecast_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    return interp_nwp_model_io.find_file(
        directory_name=directory_name,
        init_time_unix_sec=init_time_unix_sec,
        forecast_hour=forecast_hour,
        model_name=model_name,
        raise_error_if_missing=raise_error_if_missing
    )


def find_files_for_period(
        directory_name, model_name,
        first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many NetCDF files, each with NWP forecasts for one init/valid time.

    All files must pertain to the same model -- but in general, the files will
    contain different forecast hours.

    :param directory_name: Path to input directory.
    :param model_name: Name of NWP model (must be accepted by
        `nwp_model_utils.check_model_name`).
    :param first_init_time_unix_sec: First initialization time in period.
    :param last_init_time_unix_sec: Last initialization time in period.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: nwp_forecast_file_names: 1-D list of paths to NetCDF files with NWP
        forecasts, one for each pair of init/valid time.
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

    nwp_forecast_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_pattern = (
            '{0:s}/{1:s}/{2:s}_{1:s}_hour[0-9][0-9][0-9].nc'
        ).format(
            directory_name,
            time_conversion.unix_sec_to_string(
                this_init_time_unix_sec, TIME_FORMAT
            ),
            model_name
        )

        these_file_names = glob.glob(this_file_pattern)

        if raise_error_if_any_missing and len(these_file_names) == 0:
            error_string = (
                'Cannot find any files for init time {0:s}.  Expected glob '
                'pattern: "{1:s}"'
            ).format(
                time_conversion.unix_sec_to_string(
                    this_init_time_unix_sec, TIME_FORMAT
                ),
                this_file_pattern
            )

            raise ValueError(error_string)

        these_file_names.sort()
        nwp_forecast_file_names += these_file_names

    if raise_error_if_all_missing and len(nwp_forecast_file_names) == 0:
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

    return nwp_forecast_file_names


def file_name_to_init_time(nwp_forecast_file_name):
    """Parses initialization time from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    return interp_nwp_model_io.file_name_to_init_time(nwp_forecast_file_name)


def file_name_to_model_name(nwp_forecast_file_name):
    """Parses NWP-model name from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: model_name: Model name.
    """

    return interp_nwp_model_io.file_name_to_model_name(nwp_forecast_file_name)


def file_name_to_forecast_hour(nwp_forecast_file_name):
    """Parses forecast hour from name of file with NWP forecasts.

    :param nwp_forecast_file_name: File path.
    :return: forecast_hour: Forecast hour.
    """

    return interp_nwp_model_io.file_name_to_forecast_hour(
        nwp_forecast_file_name
    )


def read_file(netcdf_file_name):
    """Reads NWP forecasts from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: nwp_forecast_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)


def write_file(nwp_forecast_table_xarray, netcdf_file_name):
    """Writes NWP output (forecasts) to NetCDF file.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    interp_nwp_model_io.write_file(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        netcdf_file_name=netcdf_file_name
    )


def write_normalization_file(norm_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters for NWP data to NetCDF file.

    :param norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    norm_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF4_CLASSIC'
    )


def read_normalization_file(netcdf_file_name):
    """Reads normalization parameters for NWP data from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
