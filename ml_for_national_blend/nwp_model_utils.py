"""Helper methods for output from any NWP model."""

import os
import sys
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

ROW_DIM = 'row'
COLUMN_DIM = 'column'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'


def read_model_coords(netcdf_file_name):
    """Reads model coordinates from NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to NetCDF file.
    :return: latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :return: longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg
        east).
    """

    error_checking.assert_file_exists(netcdf_file_name)
    coord_table_xarray = xarray.open_dataset(netcdf_file_name)

    return (
        coord_table_xarray[LATITUDE_KEY].values,
        coord_table_xarray[LONGITUDE_KEY].values
    )
