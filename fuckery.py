"""Scratch space."""

import numpy
import xarray

coord_matrix = numpy.loadtxt('/home/ralager/Downloads/wrf_arw_coords.txt', delimiter=',')
print(coord_matrix[:5, :])

row_indices = numpy.round(coord_matrix[:, 0]).astype(int)
column_indices = numpy.round(coord_matrix[:, 1]).astype(int)
latitudes_deg_n = coord_matrix[:, 2]
longitudes_deg_e = coord_matrix[:, 3]

num_rows = numpy.max(row_indices)
num_columns = numpy.max(column_indices)

# row_index_matrix = numpy.reshape(row_indices, (num_rows, num_columns), order='F')
latitude_matrix_deg_n = numpy.reshape(latitudes_deg_n, (num_rows, num_columns), order='F')
longitude_matrix_deg_n = numpy.reshape(longitudes_deg_e, (num_rows, num_columns), order='F')

ROW_DIM = 'row'
COLUMN_DIM = 'column'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'

data_dict = {
    LATITUDE_KEY: ((ROW_DIM, COLUMN_DIM), latitude_matrix_deg_n),
    LONGITUDE_KEY: ((ROW_DIM, COLUMN_DIM), longitude_matrix_deg_n)
}

coord_table_xarray = xarray.Dataset(data_vars=data_dict)
print(coord_table_xarray)

coord_table_xarray.to_netcdf(path='/home/ralager/ml_for_national_blend/ml_for_national_blend/utils/wrf_arw_coords.nc', mode='w', format='NETCDF3_64BIT')
