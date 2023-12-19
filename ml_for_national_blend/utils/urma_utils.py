"""Helper methods for URMA output."""

import os
import xarray
from gewittergefahr.gg_utils import error_checking

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

VALID_TIME_DIM = 'valid_time_unix_sec'
ROW_DIM = 'row'
COLUMN_DIM = 'column'
FIELD_DIM = 'field_name'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
DATA_KEY = 'data_matrix'

TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
WIND_GUST_10METRE_NAME = 'wind_gust_10m_agl_m_s01'

ALL_FIELD_NAMES = [
    TEMPERATURE_2METRE_NAME, DEWPOINT_2METRE_NAME, U_WIND_10METRE_NAME,
    V_WIND_10METRE_NAME, WIND_GUST_10METRE_NAME
]


def check_field_name(field_name):
    """Ensures that field name is valid.

    :param field_name: String (must be in list `ALL_FIELD_NAMES`).
    :raises: ValueError: if `field_name not in ALL_FIELD_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_FIELD_NAMES:
        return

    error_string = (
        'Field name "{0:s}" is not in the list of accepted field names '
        '(below):\n{1:s}'
    ).format(
        field_name, str(ALL_FIELD_NAMES)
    )

    raise ValueError(error_string)


def read_grid_coords(netcdf_file_name=None):
    """Reads grid coordinates from NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to NetCDF file.
    :return: latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :return: longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg
        east).
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/urma_coords.nc'.format(THIS_DIRECTORY_NAME)

    error_checking.assert_file_exists(netcdf_file_name)
    coord_table_xarray = xarray.open_dataset(netcdf_file_name)

    return (
        coord_table_xarray[LATITUDE_KEY].values,
        coord_table_xarray[LONGITUDE_KEY].values
    )


def concat_over_time(urma_tables_xarray):
    """Concatenates URMA tables over valid time.

    :param urma_tables_xarray: 1-D list of xarray tables with URMA data.
    :return: urma_table_xarray: Single xarray table with URMA data.
    """

    return xarray.concat(
        urma_tables_xarray, dim=VALID_TIME_DIM, data_vars=[DATA_KEY],
        coords='minimal', compat='identical', join='exact'
    )
