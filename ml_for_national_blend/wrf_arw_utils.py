"""Helper methods for WRF-ARW model."""

import os
import sys
import warnings
import numpy
import xarray
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import error_checking
import nwp_model_utils

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

TOLERANCE = 1e-6

INIT_TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600

FORECAST_HOUR_DIM = 'forecast_hour'
ROW_DIM = 'row'
COLUMN_DIM = 'column'
FIELD_DIM = 'field_name'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
DATA_KEY = 'data_matrix'

MSL_PRESSURE_NAME = 'pressure_mean_sea_level_pascals'
SURFACE_PRESSURE_NAME = 'pressure_surface_pascals'
TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
PRECIP_NAME = 'accumulated_precip_metres'
HEIGHT_500MB_NAME = 'geopotential_height_500mb_m_asl'
HEIGHT_700MB_NAME = 'geopotential_height_700mb_m_asl'
RELATIVE_HUMIDITY_500MB_NAME = 'relative_humidity_500mb'
RELATIVE_HUMIDITY_700MB_NAME = 'relative_humidity_700mb'
RELATIVE_HUMIDITY_850MB_NAME = 'relative_humidity_850mb'
U_WIND_500MB_NAME = 'u_wind_500mb_m_s01'
U_WIND_700MB_NAME = 'u_wind_700mb_m_s01'
U_WIND_1000MB_NAME = 'u_wind_1000mb_m_s01'
V_WIND_500MB_NAME = 'v_wind_500mb_m_s01'
V_WIND_700MB_NAME = 'v_wind_700mb_m_s01'
V_WIND_1000MB_NAME = 'v_wind_1000mb_m_s01'
TEMPERATURE_850MB_NAME = 'temperature_850mb_kelvins'
TEMPERATURE_950MB_NAME = 'temperature_950mb_kelvins'
MIN_RELATIVE_HUMIDITY_2METRE_NAME = 'hourly_min_relative_humidity_2m_agl'
MAX_RELATIVE_HUMIDITY_2METRE_NAME = 'hourly_max_relative_humidity_2m_agl'

ALL_FIELD_NAMES = [
    MSL_PRESSURE_NAME, SURFACE_PRESSURE_NAME, TEMPERATURE_2METRE_NAME,
    DEWPOINT_2METRE_NAME, RELATIVE_HUMIDITY_2METRE_NAME, U_WIND_10METRE_NAME,
    V_WIND_10METRE_NAME, PRECIP_NAME, HEIGHT_500MB_NAME, HEIGHT_700MB_NAME,
    RELATIVE_HUMIDITY_500MB_NAME, RELATIVE_HUMIDITY_700MB_NAME,
    RELATIVE_HUMIDITY_850MB_NAME,
    U_WIND_500MB_NAME, U_WIND_700MB_NAME, U_WIND_1000MB_NAME,
    V_WIND_500MB_NAME, V_WIND_700MB_NAME, V_WIND_1000MB_NAME,
    TEMPERATURE_850MB_NAME, TEMPERATURE_950MB_NAME,
    MIN_RELATIVE_HUMIDITY_2METRE_NAME, MAX_RELATIVE_HUMIDITY_2METRE_NAME
]

MAYBE_MISSING_FIELD_NAMES = []
ALL_FORECAST_HOURS = numpy.linspace(1, 48, num=48, dtype=int)


def check_init_time(init_time_unix_sec):
    """Checks validity of initialization time.  Must be either 00Z or 12Z.

    :param init_time_unix_sec: Initialization time.
    :raises: AssertionError: if time is neither 00Z nor 12Z.
    """

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, INIT_TIME_FORMAT
    )

    hour_string = init_time_string.split('-')[-1]
    assert hour_string in ['00', '12']


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


def read_model_coords(netcdf_file_name=None):
    """Reads WRF-ARW grid coordinates from file.

    This is a lightweight wrapper for `nwp_model_utils.read_model_coords`.

    :param netcdf_file_name: See documentation for
        `nwp_model_utils.read_model_coords`.
    :return: latitude_matrix_deg_n: Same.
    :return: longitude_matrix_deg_e: Same.
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/wrf_arw_coords.nc'.format(THIS_DIRECTORY_NAME)

    return nwp_model_utils.read_model_coords(netcdf_file_name)


def concat_over_forecast_hours(wrf_arw_tables_xarray):
    """Concatenates WRF-ARW tables over forecast hour.

    :param wrf_arw_tables_xarray: 1-D list of xarray tables with WRF-ARW data.
    :return: wrf_arw_table_xarray: Single xarray table with WRF-ARW data.
    """

    return xarray.concat(
        wrf_arw_tables_xarray, dim=FORECAST_HOUR_DIM, data_vars=[DATA_KEY],
        coords='minimal', compat='identical', join='exact'
    )


def subset_by_row(wrf_arw_table_xarray, desired_row_indices):
    """Subsets WRF-ARW table by grid row.

    :param wrf_arw_table_xarray: xarray table with WRF-ARW forecasts.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: wrf_arw_table_xarray: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return wrf_arw_table_xarray.isel({ROW_DIM: desired_row_indices})


def subset_by_column(wrf_arw_table_xarray, desired_column_indices):
    """Subsets WRF-ARW table by grid column.

    :param wrf_arw_table_xarray: xarray table with WRF-ARW forecasts.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: wrf_arw_table_xarray: Same as input but maybe with fewer columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return wrf_arw_table_xarray.isel({COLUMN_DIM: desired_column_indices})


def subset_by_forecast_hour(wrf_arw_table_xarray, desired_forecast_hours):
    """Subsets WRF-ARW table by forecast hour.

    :param wrf_arw_table_xarray: xarray table with WRF-ARW forecasts.
    :param desired_forecast_hours: 1-D numpy array of desired forecast hours.
    :return: wrf_arw_table_xarray: Same as input but maybe with fewer forecast
        hours.
    """

    error_checking.assert_is_numpy_array(
        desired_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_forecast_hours)
    error_checking.assert_is_greater_numpy_array(desired_forecast_hours, 0)

    return wrf_arw_table_xarray.sel({FORECAST_HOUR_DIM: desired_forecast_hours})


def get_field(wrf_arw_table_xarray, field_name):
    """Extracts one field from WRF-ARW table.

    H = number of forecast hours
    M = number of rows in grid
    N = number of columns in grid

    :param wrf_arw_table_xarray: xarray table with WRF-ARW forecasts.
    :param field_name: Field name.
    :return: data_matrix: H-by-M-by-N numpy array of data values.
    """

    check_field_name(field_name)

    k = numpy.where(
        wrf_arw_table_xarray.coords[FIELD_DIM].values == field_name
    )[0][0]
    return wrf_arw_table_xarray[DATA_KEY].values[..., k]


def read_nonprecip_field_different_times(
        wrf_arw_table_xarray, init_time_unix_sec, field_name,
        valid_time_matrix_unix_sec):
    """Reads any field other than precip, with different valid time at each px.

    M = number of rows in grid
    N = number of columns in grid

    :param wrf_arw_table_xarray: xarray table with WRF-ARW data for one model
        run, in format returned by `wrf_arw_io.read_file`.
    :param init_time_unix_sec: Initialization time.
    :param field_name: Field name.
    :param valid_time_matrix_unix_sec: M-by-N numpy array of valid times.  Will
        interpolate between forecast hours where necessary.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    # Check input args.
    check_init_time(init_time_unix_sec)

    num_grid_rows = len(wrf_arw_table_xarray.coords[ROW_DIM].values)
    num_grid_columns = len(wrf_arw_table_xarray.coords[COLUMN_DIM].values)
    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)

    error_checking.assert_is_integer_numpy_array(valid_time_matrix_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_time_matrix_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(
        valid_time_matrix_unix_sec, init_time_unix_sec
    )

    # Do actual stuff.
    data_matrix_orig_fcst_hours = get_field(
        wrf_arw_table_xarray=wrf_arw_table_xarray,
        field_name=field_name
    )

    wrf_arw_valid_times_unix_sec = (
        init_time_unix_sec +
        wrf_arw_table_xarray.coords[FORECAST_HOUR_DIM].values * HOURS_TO_SECONDS
    )

    bad_hour_flags = numpy.any(
        numpy.isnan(data_matrix_orig_fcst_hours), axis=(1, 2)
    )
    good_hour_indices = numpy.where(numpy.invert(bad_hour_flags))[0]
    data_matrix_orig_fcst_hours = data_matrix_orig_fcst_hours[
        good_hour_indices, ...
    ]
    wrf_arw_valid_times_unix_sec = wrf_arw_valid_times_unix_sec[
        good_hour_indices
    ]

    unique_desired_valid_times_unix_sec = numpy.unique(
        valid_time_matrix_unix_sec
    )

    interp_object = interp1d(
        x=wrf_arw_valid_times_unix_sec,
        y=data_matrix_orig_fcst_hours,
        kind='linear', axis=0, assume_sorted=True, bounds_error=False,
        fill_value=(
            data_matrix_orig_fcst_hours[0, ...],
            data_matrix_orig_fcst_hours[-1, ...]
        )
    )
    interp_data_matrix_3d = interp_object(unique_desired_valid_times_unix_sec)

    interp_data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )

    for i in range(len(unique_desired_valid_times_unix_sec)):
        rowcol_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_desired_valid_times_unix_sec[i]
        )
        interp_data_matrix[rowcol_indices] = (
            interp_data_matrix_3d[i, ...][rowcol_indices]
        )

    assert not numpy.any(numpy.isnan(interp_data_matrix))
    return interp_data_matrix


def read_24hour_precip_different_times(
        wrf_arw_table_xarray, init_time_unix_sec, valid_time_matrix_unix_sec):
    """Reads 24-hour accumulated precip, with different valid time at each px.

    :param wrf_arw_table_xarray: See doc for
        `read_nonprecip_field_different_times`.
    :param init_time_unix_sec: Same.
    :param valid_time_matrix_unix_sec: Same.
    :return: data_matrix: Same.
    """

    # Check input args.
    check_init_time(init_time_unix_sec)

    num_grid_rows = len(wrf_arw_table_xarray.coords[ROW_DIM].values)
    num_grid_columns = len(wrf_arw_table_xarray.coords[COLUMN_DIM].values)
    expected_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)

    error_checking.assert_is_integer_numpy_array(valid_time_matrix_unix_sec)
    error_checking.assert_is_numpy_array(
        valid_time_matrix_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_geq_numpy_array(
        valid_time_matrix_unix_sec, init_time_unix_sec + 24 * HOURS_TO_SECONDS
    )

    # Do actual stuff.
    orig_full_run_precip_matrix_metres = get_field(
        wrf_arw_table_xarray=wrf_arw_table_xarray,
        field_name=PRECIP_NAME
    )
    assert not numpy.any(numpy.isnan(orig_full_run_precip_matrix_metres))

    orig_forecast_hours = wrf_arw_table_xarray.coords[FORECAST_HOUR_DIM].values

    if not numpy.all(numpy.isin(ALL_FORECAST_HOURS, orig_forecast_hours)):
        missing_hour_array_string = str(
            ALL_FORECAST_HOURS[
                numpy.isin(ALL_FORECAST_HOURS, orig_forecast_hours) == False
            ]
        )

        warning_string = (
            'POTENTIAL ERROR: Expected {0:d} forecast hours in WRF-ARW table.  '
            'Instead, got {1:d} forecast hours, with the following hours '
            'missing:\n{2:s}'
        ).format(
            len(ALL_FORECAST_HOURS),
            len(orig_forecast_hours),
            missing_hour_array_string
        )

        warnings.warn(warning_string)

        interp_object = interp1d(
            x=orig_forecast_hours,
            y=orig_full_run_precip_matrix_metres,
            kind='linear', axis=0, assume_sorted=True, bounds_error=False,
            fill_value=(
                orig_full_run_precip_matrix_metres[0, ...],
                orig_full_run_precip_matrix_metres[-1, ...]
            )
        )
        orig_full_run_precip_matrix_metres = interp_object(ALL_FORECAST_HOURS)
        orig_forecast_hours = ALL_FORECAST_HOURS + 0

    wrf_arw_valid_times_unix_sec = (
        init_time_unix_sec + orig_forecast_hours * HOURS_TO_SECONDS
    )
    unique_desired_valid_times_unix_sec = numpy.unique(
        valid_time_matrix_unix_sec
    )

    interp_object = interp1d(
        x=wrf_arw_valid_times_unix_sec,
        y=orig_full_run_precip_matrix_metres,
        kind='linear', axis=0, assume_sorted=True, bounds_error=False,
        fill_value=(
            orig_full_run_precip_matrix_metres[0, ...],
            orig_full_run_precip_matrix_metres[-1, ...]
        )
    )
    interp_end_precip_matrix_metres_3d = interp_object(
        unique_desired_valid_times_unix_sec
    )
    interp_start_precip_matrix_metres_3d = interp_object(
        unique_desired_valid_times_unix_sec - 24 * HOURS_TO_SECONDS
    )

    interp_24hour_precip_matrix_metres = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )

    for i in range(len(unique_desired_valid_times_unix_sec)):
        rowcol_indices = numpy.where(
            valid_time_matrix_unix_sec == unique_desired_valid_times_unix_sec[i]
        )
        interp_24hour_precip_matrix_metres[rowcol_indices] = (
            interp_end_precip_matrix_metres_3d[i, ...][rowcol_indices] -
            interp_start_precip_matrix_metres_3d[i, ...][rowcol_indices]
        )

    assert not numpy.any(numpy.isnan(interp_24hour_precip_matrix_metres))
    assert numpy.all(interp_24hour_precip_matrix_metres >= -1 * TOLERANCE)
    interp_24hour_precip_matrix_metres = numpy.maximum(
        interp_24hour_precip_matrix_metres, 0.
    )

    return interp_24hour_precip_matrix_metres


def precip_from_incremental_to_full_run(wrf_arw_table_xarray):
    """Converts precip from incremental values to full-run values.

    "Incremental value" = an accumulation between two forecast hours
    "Full-run value" = accumulation over the entire model run, up to a forecast
                       hour

    :param wrf_arw_table_xarray: xarray table with WRF-ARW forecasts.
    :return: wrf_arw_table_xarray: Same as input but with full-run precip.
    """

    forecast_hours = wrf_arw_table_xarray.coords[FORECAST_HOUR_DIM].values
    num_forecast_hours = len(forecast_hours)

    assert numpy.all(numpy.isin(
        element=ALL_FORECAST_HOURS,
        test_elements=forecast_hours
    ))

    data_matrix = wrf_arw_table_xarray[DATA_KEY].values

    for j in range(num_forecast_hours)[::-1]:
        addend_indices = numpy.where(forecast_hours <= forecast_hours[j])[0]

        for this_field_name in [PRECIP_NAME]:
            k = numpy.where(
                wrf_arw_table_xarray.coords[FIELD_DIM].values == this_field_name
            )[0][0]

            data_matrix[j, ..., k] = numpy.sum(
                data_matrix[addend_indices, ..., k], axis=0
            )

    wrf_arw_table_xarray = wrf_arw_table_xarray.assign({
        DATA_KEY: (
            wrf_arw_table_xarray[DATA_KEY].dims, data_matrix
        )
    })

    return wrf_arw_table_xarray


def remove_negative_precip(wrf_arw_table_xarray):
    """Removes negative precip accumulations.

    This method requires an input table with full-run values, not incremental
    values -- but it ensures that all incremental values, and thus all full-run
    values, are positive.  For the definitions of "full-run" and "incremental"
    precip, see documentation for `precip_from_incremental_to_full_run`.

    :param wrf_arw_table_xarray: xarray table with WRF-ARW forecasts.
    :return: wrf_arw_table_xarray: Same as input but without negative precip
        accumulations.
    """

    forecast_hours = wrf_arw_table_xarray.coords[FORECAST_HOUR_DIM].values
    num_forecast_hours = len(forecast_hours)

    data_matrix = wrf_arw_table_xarray[DATA_KEY].values

    for j in range(num_forecast_hours):
        if j == 0:
            continue

        for this_field_name in [PRECIP_NAME]:
            ks = numpy.where(
                wrf_arw_table_xarray.coords[FIELD_DIM].values == this_field_name
            )[0]

            if len(ks) == 0:
                continue

            k = ks[0]
            data_matrix[j, ..., k] = numpy.nanmax(
                data_matrix[:(j + 1), ..., k], axis=0
            )

    wrf_arw_table_xarray = wrf_arw_table_xarray.assign({
        DATA_KEY: (
            wrf_arw_table_xarray[DATA_KEY].dims, data_matrix
        )
    })

    return wrf_arw_table_xarray
