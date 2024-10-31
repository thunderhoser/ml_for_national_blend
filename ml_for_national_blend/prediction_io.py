"""Input/output methods for predictions."""

import os
import sys
import numpy
import xarray
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking

TOLERANCE = 1e-6
TIME_FORMAT = '%Y-%m-%d-%H'

HOURS_TO_SECONDS = 3600
DEFAULT_INIT_TIME_INTERVAL_SEC = 6 * HOURS_TO_SECONDS  # TODO(thunderhoser): This will change.

ROW_DIM = 'grid_row'
COLUMN_DIM = 'grid_column'
FIELD_DIM = 'field'
FIELD_CHAR_DIM = 'field_char'
ENSEMBLE_MEMBER_DIM = 'ensemble_member'
DUMMY_ENSEMBLE_MEMBER_DIM = 'dummy_ensemble_member'

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILES_KEY = 'isotonic_model_file_names'
UNCERTAINTY_CALIB_MODEL_FILES_KEY = 'uncertainty_calib_model_file_names'
INIT_TIME_KEY = 'init_time_unix_sec'

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


def find_files_for_period(
        directory_name, first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Finds files with predictions over a time period, one per init time.

    :param directory_name: Path to input directory.
    :param first_init_time_unix_sec: First init time in period.
    :param last_init_time_unix_sec: Last init time in period.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: prediction_file_names: 1-D list of paths to NetCDF files with
        predictions, one per daily model run.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=DEFAULT_INIT_TIME_INTERVAL_SEC,
        include_endpoint=True
    )

    prediction_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_name = find_file(
            directory_name=directory_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isfile(this_file_name):
            prediction_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(prediction_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from times {1:s} to '
            '{2:s}.'
        ).format(
            directory_name,
            time_conversion.unix_sec_to_string(
                first_init_time_unix_sec, TIME_FORMAT
            ),
            time_conversion.unix_sec_to_string(
                last_init_time_unix_sec, TIME_FORMAT
            )
        )
        raise ValueError(error_string)

    return prediction_file_names


def find_rap_based_files_for_period(
        directory_name, first_init_time_unix_sec, last_init_time_unix_sec,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True):
    """Same as `find_files_for_period` but finds RAP-based forecasts.

    :param directory_name: See doc for `find_files_for_period`.
    :param first_init_time_unix_sec: Same.
    :param last_init_time_unix_sec: Same.
    :param raise_error_if_any_missing: Same.
    :param raise_error_if_all_missing: Same.
    :return: prediction_file_names: Same.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=3 * HOURS_TO_SECONDS,
        include_endpoint=True
    )

    good_indices = numpy.where(
        numpy.invert(numpy.mod(init_times_unix_sec, 6 * HOURS_TO_SECONDS) == 0)
    )[0]
    init_times_unix_sec = init_times_unix_sec[good_indices]

    prediction_file_names = []

    for this_init_time_unix_sec in init_times_unix_sec:
        this_file_name = find_file(
            directory_name=directory_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if os.path.isfile(this_file_name):
            prediction_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(prediction_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from times {1:s} to '
            '{2:s}.'
        ).format(
            directory_name,
            time_conversion.unix_sec_to_string(
                first_init_time_unix_sec, TIME_FORMAT
            ),
            time_conversion.unix_sec_to_string(
                last_init_time_unix_sec, TIME_FORMAT
            )
        )
        raise ValueError(error_string)

    return prediction_file_names


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
    ptx = prediction_table_xarray

    target_field_names = [
        f.decode('utf-8') for f in ptx[FIELD_NAME_KEY].values
    ]
    ptx = ptx.assign({
        FIELD_NAME_KEY: (ptx[FIELD_NAME_KEY].dims, target_field_names)
    })

    if ISOTONIC_MODEL_FILES_KEY not in ptx.attrs:
        ptx.attrs[ISOTONIC_MODEL_FILES_KEY] = ''
    if UNCERTAINTY_CALIB_MODEL_FILES_KEY not in ptx.attrs:
        ptx.attrs[UNCERTAINTY_CALIB_MODEL_FILES_KEY] = ''

    if ptx.attrs[ISOTONIC_MODEL_FILES_KEY] == '':
        ptx.attrs[ISOTONIC_MODEL_FILES_KEY] = None
    else:
        ptx.attrs[ISOTONIC_MODEL_FILES_KEY] = (
            ptx.attrs[ISOTONIC_MODEL_FILES_KEY].split(' ')
        )

    if ptx.attrs[UNCERTAINTY_CALIB_MODEL_FILES_KEY] == '':
        ptx.attrs[UNCERTAINTY_CALIB_MODEL_FILES_KEY] = None
    else:
        ptx.attrs[UNCERTAINTY_CALIB_MODEL_FILES_KEY] = (
            ptx.attrs[UNCERTAINTY_CALIB_MODEL_FILES_KEY].split(' ')
        )

    if INIT_TIME_KEY not in ptx.data_vars:
        return ptx

    init_times_unix_sec = ptx[INIT_TIME_KEY].values
    assert len(init_times_unix_sec) == 1

    target_matrix = ptx[TARGET_KEY].values[0, ...]
    prediction_matrix = numpy.expand_dims(
        ptx[PREDICTION_KEY].values[0, ...], axis=-1
    )
    latitude_matrix_deg_n = ptx[LATITUDE_KEY].values[0, ...]
    longitude_matrix_deg_e = ptx[LONGITUDE_KEY].values[0, ...]

    main_data_dict = {
        TARGET_KEY: (
            (ROW_DIM, COLUMN_DIM, FIELD_DIM),
            target_matrix
        ),
        PREDICTION_KEY: (
            (ROW_DIM, COLUMN_DIM, FIELD_DIM, ENSEMBLE_MEMBER_DIM),
            prediction_matrix
        ),
        LATITUDE_KEY: (
            (ROW_DIM, COLUMN_DIM),
            latitude_matrix_deg_n
        ),
        LONGITUDE_KEY: (
            (ROW_DIM, COLUMN_DIM),
            longitude_matrix_deg_e
        ),
        FIELD_NAME_KEY: (
            (FIELD_DIM,),
            target_field_names
        )
    }

    attribute_dict = {
        MODEL_FILE_KEY: ptx.attrs[MODEL_FILE_KEY],
        ISOTONIC_MODEL_FILES_KEY: ptx.attrs[ISOTONIC_MODEL_FILES_KEY],
        UNCERTAINTY_CALIB_MODEL_FILES_KEY:
            ptx.attrs[UNCERTAINTY_CALIB_MODEL_FILES_KEY],
        INIT_TIME_KEY: int(numpy.round(init_times_unix_sec[0]))
    }

    return xarray.Dataset(data_vars=main_data_dict, attrs=attribute_dict)


def write_file(
        netcdf_file_name, target_matrix, prediction_matrix,
        latitude_matrix_deg_n, longitude_matrix_deg_e, field_names,
        init_time_unix_sec, model_file_name, isotonic_model_file_names,
        uncertainty_calib_model_file_names):
    """Writes predictions to NetCDF file.

    M = number of rows in grid
    N = number of columns in grid
    F = number of target fields
    S = number of ensemble members

    :param netcdf_file_name: Path to output file.
    :param target_matrix: M-by-N-by-F numpy array of actual values.
    :param prediction_matrix: M-by-N-by-F-by-S numpy array of predicted values.
    :param latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :param longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg east).
    :param field_names: length-F list of field names.
    :param init_time_unix_sec: Initialization time.
    :param model_file_name: Path to file with trained model.
    :param isotonic_model_file_names: 1-D list of paths to files with isotonic-
        regression models (one per target field), used to bias-correct ensemble
        means.  If N/A, make this None.
    :param uncertainty_calib_model_file_names: 1-D list of paths to files with
        uncertainty-calibration models (one per target field), used to bias-
        correct ensemble spreads.  If N/A, make this None.
    """

    # Check input args.
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_string(model_file_name)

    error_checking.assert_is_numpy_array(target_matrix, num_dimensions=3)
    # error_checking.assert_is_numpy_array_without_nan(target_matrix)

    num_rows = target_matrix.shape[0]
    num_columns = target_matrix.shape[1]
    num_fields = target_matrix.shape[2]

    if isotonic_model_file_names is not None:
        error_checking.assert_is_string_list(isotonic_model_file_names)
        error_checking.assert_is_numpy_array(
            numpy.array(isotonic_model_file_names),
            exact_dimensions=numpy.array([num_fields], dtype=int)
        )

    if uncertainty_calib_model_file_names is not None:
        error_checking.assert_is_string_list(uncertainty_calib_model_file_names)
        error_checking.assert_is_numpy_array(
            numpy.array(uncertainty_calib_model_file_names),
            exact_dimensions=numpy.array([num_fields], dtype=int)
        )

    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=4)
    ensemble_size = prediction_matrix.shape[3]
    expected_dim = numpy.array(
        [num_rows, num_columns, num_fields, ensemble_size], dtype=int
    )
    error_checking.assert_is_numpy_array(
        prediction_matrix, exact_dimensions=expected_dim
    )
    # error_checking.assert_is_numpy_array_without_nan(prediction_matrix)

    expected_dim = numpy.array([num_rows, num_columns], dtype=int)
    error_checking.assert_is_numpy_array(
        latitude_matrix_deg_n, exact_dimensions=expected_dim
    )
    error_checking.assert_is_valid_lat_numpy_array(
        latitude_matrix_deg_n, allow_nan=False
    )

    error_checking.assert_is_numpy_array(
        longitude_matrix_deg_e, exact_dimensions=expected_dim
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
        netcdf_file_name, 'w', format='NETCDF4_CLASSIC'
    )

    num_field_chars = max([len(f) for f in field_names])

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(
        ISOTONIC_MODEL_FILES_KEY,
        '' if isotonic_model_file_names is None
        else ' '.join(isotonic_model_file_names)
    )
    dataset_object.setncattr(
        UNCERTAINTY_CALIB_MODEL_FILES_KEY,
        '' if uncertainty_calib_model_file_names is None
        else ' '.join(uncertainty_calib_model_file_names)
    )
    dataset_object.setncattr(INIT_TIME_KEY, init_time_unix_sec)
    dataset_object.createDimension(ROW_DIM, num_rows)
    dataset_object.createDimension(COLUMN_DIM, num_columns)
    dataset_object.createDimension(FIELD_DIM, num_fields)
    dataset_object.createDimension(FIELD_CHAR_DIM, num_field_chars)
    dataset_object.createDimension(ENSEMBLE_MEMBER_DIM, ensemble_size)

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM)
    dataset_object.createVariable(
        TARGET_KEY, datatype=numpy.float64, dimensions=these_dim
    )
    dataset_object.variables[TARGET_KEY][:] = target_matrix

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, ENSEMBLE_MEMBER_DIM)
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


def take_ensemble_mean(prediction_table_xarray):
    """Takes ensemble mean for each atomic example.

    One atomic example = one init time, one valid time, one field, one pixel

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with only one prediction (the
        ensemble mean) per atomic example.
    """

    ptx = prediction_table_xarray
    ensemble_size = len(ptx.coords[ENSEMBLE_MEMBER_DIM].values)
    if ensemble_size == 1:
        return ptx

    ptx = ptx.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
    })

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, DUMMY_ENSEMBLE_MEMBER_DIM)

    ptx = ptx.assign({
        PREDICTION_KEY: (
            these_dim,
            numpy.mean(ptx[PREDICTION_KEY].values, axis=-1, keepdims=True)
        )
    })

    ptx = ptx.rename({DUMMY_ENSEMBLE_MEMBER_DIM: ENSEMBLE_MEMBER_DIM})
    prediction_table_xarray = ptx
    return prediction_table_xarray


def prep_for_uncertainty_calib_training(prediction_table_xarray):
    """Prepares predictions to train uncertainty calibration.

    Specifically, for every atomic example, this method replaces the full
    ensemble with the ensemble variance and replaces the target with the squared
    error of the ensemble mean.

    One atomic example = one init time, one valid time, one field, one pixel

    :param prediction_table_xarray: xarray table in format returned by
        `prediction_io.read_file`.
    :return: prediction_table_xarray: Same but with the aforementioned changes.
    """

    ptx = prediction_table_xarray
    ensemble_size = len(ptx.coords[ENSEMBLE_MEMBER_DIM].values)
    assert ensemble_size > 1

    prediction_variance_matrix = numpy.var(
        ptx[PREDICTION_KEY].values, axis=-1, ddof=1, keepdims=True
    )
    squared_error_matrix = (
        numpy.mean(ptx[PREDICTION_KEY].values, axis=-1) -
        ptx[TARGET_KEY].values
    ) ** 2

    ptx = ptx.assign_coords({
        DUMMY_ENSEMBLE_MEMBER_DIM: numpy.array([0], dtype=int)
    })

    these_dim = (ROW_DIM, COLUMN_DIM, FIELD_DIM, DUMMY_ENSEMBLE_MEMBER_DIM)

    ptx = ptx.assign({
        PREDICTION_KEY: (
            these_dim,
            prediction_variance_matrix
        ),
        TARGET_KEY:(
            (ROW_DIM, COLUMN_DIM, FIELD_DIM),
            squared_error_matrix
        )
    })

    ptx = ptx.rename({DUMMY_ENSEMBLE_MEMBER_DIM: ENSEMBLE_MEMBER_DIM})
    prediction_table_xarray = ptx
    return prediction_table_xarray
