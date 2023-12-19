"""Input/output methods for processed URMA data."""

import os
import sys
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import error_checking

DATE_FORMAT = '%Y%m%d'


def find_file(directory_name, valid_date_string, raise_error_if_missing=True):
    """Finds NetCDF file with URMA data for one day.

    :param directory_name: Path to input directory.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: urma_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    urma_file_name = '{0:s}/urma_{1:s}.nc'.format(
        directory_name, valid_date_string
    )

    if raise_error_if_missing and not os.path.isfile(urma_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            urma_file_name
        )
        raise ValueError(error_string)

    return urma_file_name


def find_files_for_period(
        directory_name, first_date_string, last_date_string,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds many NetCDF files, each with URMA output for one day.

    :param directory_name: Path to input directory.
    :param first_date_string: First date (format "yyyymmdd") in period.
    :param last_date_string: Last date (format "yyyymmdd") in period.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: netcdf_file_names: 1-D list of paths to NetCDF files with URMA
        output, one per day.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    netcdf_file_names = []

    for this_date_string in valid_date_strings:
        this_file_name = find_file(
            directory_name=directory_name,
            valid_date_string=this_date_string,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isfile(this_file_name):
            netcdf_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(netcdf_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from valid dates {1:s} '
            'to {2:s}.'
        ).format(
            directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return netcdf_file_names


def file_name_to_date(urma_file_name):
    """Parses valid date from name of file with URMA outputs.

    :param urma_file_name: File path.
    :return: valid_date_string: Valid date.
    """

    pathless_file_name = os.path.split(urma_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    valid_date_string = extensionless_file_name.split('_')[-1]

    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    return valid_date_string


def read_file(netcdf_file_name):
    """Reads URMA output from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: urma_table_xarray: xarray table.  Documentation in the xarray table
        should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(urma_table_xarray, netcdf_file_name):
    """Writes URMA output to NetCDF file.

    :param urma_table_xarray: xarray table in format returned by `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    urma_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )
