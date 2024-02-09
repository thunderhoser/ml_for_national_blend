"""Helper methods for National Blend of Models (NBM)."""

import os
import sys
import numpy
import xarray
import pyproj
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'

NBM_PROJECTION_OBJECT = pyproj.Proj(
    proj='lcc', lat_1=25., lat_2=25., lon_0=265.,
    R=6371200., ellps='sphere',
    x_0=3271151.6058371766, y_0=-2604259.810222088
)

NBM_X_COORDS_METRES = numpy.linspace(0, 5.95306383e+06, num=2345, dtype=float)
NBM_Y_COORDS_METRES = numpy.linspace(0, 4.05336599e+06, num=1597, dtype=float)


def read_coords(netcdf_file_name=None):
    """Reads NBM lat-long coordinates from NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to NetCDF file.
    :return: latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :return: longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg
        east).
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/nbm_coords.nc'.format(THIS_DIRECTORY_NAME)

    error_checking.assert_file_exists(netcdf_file_name)
    coord_table_xarray = xarray.open_dataset(netcdf_file_name)

    return (
        coord_table_xarray[LATITUDE_KEY].values,
        coord_table_xarray[LONGITUDE_KEY].values
    )


def project_latlng_to_xy(latitudes_deg_n, longitudes_deg_e):
    """Converts points from lat-long to x-y coordinates...

    with the x-y coordinates being in the NBM's projection space (Lambert
    conformal).

    :param latitudes_deg_n: numpy array of latitudes (deg north).
    :param longitudes_deg_e: numpy array of longitudes (deg east) with the same
        shape.
    :return: x_coords_metres: numpy array of x-coords with the same shape.
    :return: y_coords_metres: numpy array of y-coords with the same shape.
    """

    error_checking.assert_is_valid_lat_numpy_array(latitudes_deg_n)
    error_checking.assert_is_valid_lng_numpy_array(longitudes_deg_e)
    error_checking.assert_is_numpy_array(
        longitudes_deg_e,
        exact_dimensions=numpy.array(latitudes_deg_n.shape, dtype=int)
    )

    x_coords_metres, y_coords_metres = NBM_PROJECTION_OBJECT(
        longitudes_deg_e, latitudes_deg_n
    )

    return x_coords_metres, y_coords_metres


def project_xy_to_latlng(x_coords_metres, y_coords_metres):
    """Converts points from x-y to lat-long coordinates...

    with the x-y coordinates being in the NBM's projection space (Lambert
    conformal).

    :param x_coords_metres: numpy array of x-coords.
    :param y_coords_metres: numpy array of y-coords with the same shape.
    :return: latitudes_deg_n: numpy array of latitudes with the same shape (deg
        north).
    :return: longitudes_deg_e: numpy array of longitudes (deg east) with the
        same shape.
    """

    error_checking.assert_is_numpy_array_without_nan(x_coords_metres)
    error_checking.assert_is_numpy_array_without_nan(y_coords_metres)
    error_checking.assert_is_numpy_array(
        y_coords_metres,
        exact_dimensions=numpy.array(x_coords_metres.shape, dtype=int)
    )

    longitudes_deg_e, latitudes_deg_n = NBM_PROJECTION_OBJECT(
        x_coords_metres, y_coords_metres, inverse=True
    )

    return latitudes_deg_n, longitudes_deg_e


def interp_data_to_nbm_grid(
        data_matrix, x_coord_matrix, y_coord_matrix, use_nearest_neigh,
        test_mode=False, new_x_coords=None, new_y_coords=None):
    """Interpolates data to NBM grid.

    m = number of rows in original grid
    n = number of columns in original grid
    C = number of channels (variables)
    M = number of rows in NBM grid
    N = number of columns in NBM grid

    :param data_matrix: m-by-n-by-C numpy array of data on original grid.
    :param x_coord_matrix: m-by-n numpy array of x-coordinates in original grid.
    :param y_coord_matrix: m-by-n numpy array of y-coordinates in original grid.
    :param use_nearest_neigh: Boolean flag.  If True (False), will use nearest-
        neighbour (linear) interpolation.
    :param test_mode: Leave this alone.
    :param new_x_coords: Leave this alone.
    :param new_y_coords: Leave this alone.
    :return: interp_data_matrix: M-by-N-by-C numpy array of data on NBM grid.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(x_coord_matrix)
    error_checking.assert_is_numpy_array(x_coord_matrix, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(y_coord_matrix)
    error_checking.assert_is_numpy_array(
        y_coord_matrix,
        exact_dimensions=numpy.array(x_coord_matrix.shape, dtype=int)
    )

    # error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=3)
    expected_dim = numpy.array(
        x_coord_matrix.shape + (data_matrix.shape[2],), dtype=int
    )
    error_checking.assert_is_numpy_array(
        data_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_boolean(use_nearest_neigh)
    error_checking.assert_is_boolean(test_mode)

    # Do actual stuff.
    point_matrix = numpy.transpose(numpy.vstack([
        numpy.ravel(y_coord_matrix), numpy.ravel(x_coord_matrix)
    ]))
    value_matrix = numpy.reshape(
        data_matrix,
        (x_coord_matrix.size, data_matrix.shape[-1])
    )

    linear_interp_object = LinearNDInterpolator(
        points=point_matrix, values=value_matrix, fill_value=numpy.nan
    )

    if test_mode:
        new_x_matrix, new_y_matrix = numpy.meshgrid(new_x_coords, new_y_coords)
    else:
        new_x_matrix, new_y_matrix = numpy.meshgrid(
            NBM_X_COORDS_METRES, NBM_Y_COORDS_METRES
        )

    interp_data_matrix = linear_interp_object(new_y_matrix, new_x_matrix)
    if not use_nearest_neigh:
        return interp_data_matrix

    nn_interp_object = NearestNDInterpolator(x=point_matrix, y=value_matrix)
    nn_interp_data_matrix = nn_interp_object(new_y_matrix, new_x_matrix)
    nn_interp_data_matrix[numpy.isnan(interp_data_matrix)] = numpy.nan
    return nn_interp_data_matrix
