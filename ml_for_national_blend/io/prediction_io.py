"""Input/output methods for predictions."""

import os
import numpy
import xarray
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H'

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
FIELD_DIM = 'field'
FIELD_CHAR_DIM = 'field_char'

MODEL_FILE_KEY = 'model_file_name'
INIT_TIME_KEY = 'init_date_unix_sec'

TARGET_KEY = 'target'
PREDICTION_KEY = 'prediction'
LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
FIELD_NAME_KEY = 'field_name'


def find_file(directory_name, init_time_unix_sec, raise_error_if_missing=True):
    """Finds NetCDF file with predictions initialized at one time.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: prediction_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT
    )
    error_checking.assert_is_boolean(raise_error_if_missing)

    prediction_file_name = '{0:s}/predictions_{1:s}.nc'.format(
        directory_name, init_time_string
    )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name
        )
        raise ValueError(error_string)

    return prediction_file_name


def file_name_to_init_time(prediction_file_name):
    """Parses initialization time from name of prediction file.

    :param prediction_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    pathless_file_name = os.path.split(prediction_file_name)[1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    init_time_string = extensionless_file_name.split('_')[1]

    return time_conversion.string_to_unix_sec(init_time_string, TIME_FORMAT)


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_table_xarray: xarray table.  Documentation in the xarray
        table should make values self-explanatory.
    """

    prediction_table_xarray = xarray.open_dataset(netcdf_file_name)
    target_field_names = [
        f.decode('utf-8') for f in
        prediction_table_xarray[FIELD_NAME_KEY].values
    ]

    return prediction_table_xarray.assign({
        FIELD_NAME_KEY: (
            prediction_table_xarray[FIELD_NAME_KEY].dims,
            target_field_names
        )
    })


def write_file(
        netcdf_file_name, target_matrix, prediction_matrix,
        latitude_matrix_deg_n, longitude_matrix_deg_e, field_names,
        init_time_unix_sec, model_file_name):
    """Writes predictions to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    T = number of target fields

    :param netcdf_file_name: Path to output file.
    :param target_matrix: M-by-N-by-T numpy array of actual values.
    :param prediction_matrix: M-by-N-by-T numpy array of predicted values.
    :param latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :param longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg east).
    :param field_names: length-T list of field names.
    :param init_time_unix_sec: Initialization time.
    :param model_file_name: Path to file with trained model.
    """

    # Check input args.
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_string(model_file_name)

    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array_without_nan(target_matrix)

    num_rows = target_matrix.shape[0]
    num_columns = target_matrix.shape[1]
    num_fields = target_matrix.shape[2]

    error_checking.assert_is_numpy_array(
        prediction_matrix,
        exact_dimensions=numpy.array(target_matrix.shape, dtype=int)
    )
    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)

    error_checking.assert_is_numpy_array(
        latitude_matrix_deg_n,
        exact_dimensions=numpy.array([num_rows, num_columns], dtype=int)
    )
    error_checking.assert_is_valid_lat_numpy_array(
        latitude_matrix_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        longitude_matrix_deg_e,
        exact_dimensions=numpy.array([num_rows, num_columns], dtype=int)
    )
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e, allow_nan=False
    )

    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_numpy_array(
        numpy.array(field_names),
        exact_dimensions=numpy.array([num_fields], dtype=int)
    )

    # Do actual stuff.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    num_field_chars = max([len(f) for f in field_names])

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(INIT_TIME_KEY, init_time_unix_sec)
    dataset_object.createDimension(ROW_DIM, num_rows)
    dataset_object.createDimension(COLUMN_DIM, num_columns)
    dataset_object.createDimension(FIELD_DIM, num_fields)
    dataset_object.createDimension(FIELD_CHAR_DIM, num_field_chars)

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM)
    dataset_object.createVariable(
        TARGET_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[TARGET_KEY][:] = target_matrix

    dataset_object.createVariable(
        PREDICTION_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[PREDICTION_KEY][:] = prediction_matrix

    these_dim = (ROW_DIM, COLUMN_DIM)
    dataset_object.createVariable(
        LATITUDE_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[LATITUDE_KEY][:] = latitude_matrix_deg_n

    dataset_object.createVariable(
        LONGITUDE_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[LONGITUDE_KEY][:] = longitude_matrix_deg_e

    this_string_format = 'S{0:d}'.format(num_field_chars)
    field_names_char_array = netCDF4.stringtochar(numpy.array(
        field_names, dtype=this_string_format
    ))

    dataset_object.createVariable(
        FIELD_NAME_KEY, datatype='S1', dimensions=(FIELD_DIM, FIELD_CHAR_DIM)
    )
    dataset_object.variables[FIELD_NAME_KEY][:] = numpy.array(
        field_names_char_array
    )

    dataset_object.close()