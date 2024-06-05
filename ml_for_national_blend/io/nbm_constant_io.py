"""Input/output methods for processed NBM constants.

NBM = National Blend of Models

All constants should be in a single NetCDF file.
"""

import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking


def read_file(netcdf_file_name):
    """Reads NBM constants from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: nbm_constant_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)


def write_file(nbm_constant_table_xarray, netcdf_file_name):
    """Writes NBM constants to NetCDF file.

    :param nbm_constant_table_xarray: xarray table in format returned by
        `read_file`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    nbm_constant_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF4_CLASSIC'
    )


def write_normalization_file(norm_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters for NBM constants to NetCDF file.

    :param norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    norm_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF4_CLASSIC'
    )


def read_normalization_file(netcdf_file_name):
    """Reads normalization parameters for NBM constants from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: norm_param_table_xarray: xarray table (metadata and variable names
        should make the table self-explanatory).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
