"""Input/output methods for region masks."""

import numpy
import xarray
import netCDF4
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
REGION_MASK_KEY = 'region_mask'


def write_file(netcdf_file_name, mask_matrix):
    """Writes region mask to NetCDF file.

    M = number of rows in NBM grid
    N = number of columns in NBM grid

    :param netcdf_file_name: Path to output file.
    :param mask_matrix: M-by-N numpy array of Boolean flags, where True (False)
        means that the pixel is (not) inside the region.
    """

    nbm_latitude_matrix_deg_n = nbm_utils.read_coords()[0]
    num_grid_rows = nbm_latitude_matrix_deg_n.shape[0]
    num_grid_columns = nbm_latitude_matrix_deg_n.shape[1]

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(
        mask_matrix,
        exact_dimensions=numpy.array(
            [num_grid_rows, num_grid_columns], dtype=int
        )
    )

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF4'
    )

    dataset_object.createDimension(ROW_DIM, num_grid_rows)
    dataset_object.createDimension(COLUMN_DIM, num_grid_columns)

    these_dim = (ROW_DIM, COLUMN_DIM)
    dataset_object.createVariable(
        REGION_MASK_KEY, datatype=numpy.int32, dimensions=these_dim
    )
    dataset_object.variables[REGION_MASK_KEY][:] = mask_matrix.astype(int)

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads region mask from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: mask_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    mask_table_xarray = xarray.open_dataset(netcdf_file_name)

    return mask_table_xarray.assign({
        REGION_MASK_KEY: (
            mask_table_xarray[REGION_MASK_KEY].dims,
            mask_table_xarray[REGION_MASK_KEY].values.astype(bool)
        )
    })
