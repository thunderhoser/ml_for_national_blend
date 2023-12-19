"""Helper methods for output from any NWP model."""

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

TOLERANCE = 1e-6

HOURS_TO_SECONDS = 3600
INIT_TIME_FORMAT = '%Y-%m-%d-%H'

FORECAST_HOUR_DIM = 'forecast_hour'
ROW_DIM = 'row'
COLUMN_DIM = 'column'
FIELD_DIM = 'field_name'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
DATA_KEY = 'data_matrix'

WRF_ARW_MODEL_NAME = 'wrf_arw'
NAM_MODEL_NAME = 'nam'
NAM_NEST_MODEL_NAME = 'nam_nest'
RAP_MODEL_NAME = 'rap'
ALL_MODEL_NAMES = [
    WRF_ARW_MODEL_NAME, NAM_MODEL_NAME, NAM_NEST_MODEL_NAME, RAP_MODEL_NAME
]

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


def check_model_name(model_name):
    """Ensures that model name is valid.

    :param model_name: String (must be in list `ALL_MODEL_NAMES`).
    :raises: ValueError: if `model_name not in ALL_MODEL_NAMES`.
    """

    error_checking.assert_is_string(model_name)
    if model_name in ALL_MODEL_NAMES:
        return

    error_string = (
        'Model name "{0:s}" is not in the list of accepted model names '
        '(below):\n{1:s}'
    ).format(
        model_name, str(ALL_MODEL_NAMES)
    )

    raise ValueError(error_string)


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


def check_init_time(init_time_unix_sec, model_name):
    """Checks validity of initialization time.

    :param init_time_unix_sec: Initialization time.
    :raises: AssertionError: if time is neither 00Z nor 12Z.
    """

    check_model_name(model_name)

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, INIT_TIME_FORMAT
    )
    hour_string = init_time_string.split('-')[-1]

    if model_name == RAP_MODEL_NAME:
        return

    if model_name == WRF_ARW_MODEL_NAME:
        assert hour_string in ['00', '12']
    else:
        assert hour_string in ['00', '06', '12', '18']


def model_to_init_time_interval(model_name):
    """Returns time interval between successive model runs.

    :param model_name: Name of model.
    :return: init_time_interval_sec: Time interval between successive model
        runs.
    """

    check_model_name(model_name)
    if model_name == RAP_MODEL_NAME:
        return HOURS_TO_SECONDS
    if model_name == WRF_ARW_MODEL_NAME:
        return 12 * HOURS_TO_SECONDS

    return 6 * HOURS_TO_SECONDS


def model_to_forecast_hours(model_name, init_time_unix_sec):
    """Returns list of all available forecast hours for the given model.

    :param model_name: Name of model.
    :param init_time_unix_sec: Initialization time.
    :return: all_forecast_hours: 1-D numpy array of integers.
    """

    check_model_name(model_name)

    if model_name == RAP_MODEL_NAME:
        init_time_string = time_conversion.unix_sec_to_string(
            init_time_unix_sec, '%Y-%m-%d-%H'
        )
        init_hour_string = init_time_string.split('-')[-1]

        if init_hour_string in ['03', '09', '15', '21']:
            return numpy.linspace(1, 51, num=51, dtype=int)

        return numpy.linspace(1, 21, num=21, dtype=int)

    if model_name == NAM_MODEL_NAME:
        return numpy.linspace(51, 84, num=12, dtype=int)

    return numpy.linspace(1, 48, num=48, dtype=int)


def model_to_maybe_missing_fields(model_name):
    """Returns list of acceptably missing fields for the given model.

    :param model_name: Name of model.
    :return: maybe_missing_field_names: 1-D list with names of acceptably
        missing fields.
    """

    check_model_name(model_name)

    if model_name == RAP_MODEL_NAME:
        return [
            MIN_RELATIVE_HUMIDITY_2METRE_NAME, MAX_RELATIVE_HUMIDITY_2METRE_NAME
        ]

    return []


def read_model_coords(netcdf_file_name=None, model_name=None):
    """Reads model coordinates from NetCDF file.

    M = number of rows in grid
    N = number of columns in grid

    :param netcdf_file_name: Path to NetCDF file.
    :param model_name: Name of model.
    :return: latitude_matrix_deg_n: M-by-N numpy array of latitudes (deg north).
    :return: longitude_matrix_deg_e: M-by-N numpy array of longitudes (deg
        east).
    """

    if netcdf_file_name is None:
        check_model_name(model_name)
        netcdf_file_name = '{0:s}/{1:s}_coords.nc'.format(
            THIS_DIRECTORY_NAME, model_name
        )

    error_checking.assert_file_exists(netcdf_file_name)
    coord_table_xarray = xarray.open_dataset(netcdf_file_name)

    return (
        coord_table_xarray[LATITUDE_KEY].values,
        coord_table_xarray[LONGITUDE_KEY].values
    )


def concat_over_forecast_hours(nwp_forecast_tables_xarray):
    """Concatenates NWP tables over forecast hour.

    :param nwp_forecast_tables_xarray: 1-D list of xarray tables with NWP
        forecasts.
    :return: nwp_forecast_table_xarray: Single xarray table with NWP forecasts.
    """

    return xarray.concat(
        nwp_forecast_tables_xarray, dim=FORECAST_HOUR_DIM, data_vars=[DATA_KEY],
        coords='minimal', compat='identical', join='exact'
    )


def subset_by_row(nwp_forecast_table_xarray, desired_row_indices):
    """Subsets NWP table by grid row.

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param desired_row_indices: 1-D numpy array with indices of desired rows.
    :return: nwp_forecast_table_xarray: Same as input but maybe with fewer rows.
    """

    error_checking.assert_is_numpy_array(desired_row_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)

    return nwp_forecast_table_xarray.isel({ROW_DIM: desired_row_indices})


def subset_by_column(nwp_forecast_table_xarray, desired_column_indices):
    """Subsets NWP table by grid column.

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param desired_column_indices: 1-D numpy array with indices of desired
        columns.
    :return: nwp_forecast_table_xarray: Same as input but maybe with fewer
        columns.
    """

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)

    return nwp_forecast_table_xarray.isel({COLUMN_DIM: desired_column_indices})


def subset_by_forecast_hour(nwp_forecast_table_xarray, desired_forecast_hours):
    """Subsets NWP table by forecast hour.

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param desired_forecast_hours: 1-D numpy array of desired forecast hours.
    :return: nwp_forecast_table_xarray: Same as input but maybe with fewer
        forecast hours.
    """

    error_checking.assert_is_numpy_array(
        desired_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_forecast_hours)
    error_checking.assert_is_greater_numpy_array(desired_forecast_hours, 0)

    return nwp_forecast_table_xarray.sel(
        {FORECAST_HOUR_DIM: desired_forecast_hours}
    )


def get_field(nwp_forecast_table_xarray, field_name):
    """Extracts one field from NWP table.

    H = number of forecast hours
    M = number of rows in grid
    N = number of columns in grid

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param field_name: Field name.
    :return: data_matrix: H-by-M-by-N numpy array of data values.
    """

    check_field_name(field_name)

    k = numpy.where(
        nwp_forecast_table_xarray.coords[FIELD_DIM].values == field_name
    )[0][0]
    return nwp_forecast_table_xarray[DATA_KEY].values[..., k]


def read_nonprecip_field_different_times(
        nwp_forecast_table_xarray, model_name, init_time_unix_sec, field_name,
        valid_time_matrix_unix_sec):
    """Reads any field other than precip, with different valid time at each px.

    M = number of rows in grid
    N = number of columns in grid

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts for one
        model run, in format returned by `nwp_model_io.read_file`.
    :param model_name: Name of model.
    :param init_time_unix_sec: Initialization time.
    :param field_name: Field name.
    :param valid_time_matrix_unix_sec: M-by-N numpy array of valid times.  Will
        interpolate between forecast hours where necessary.
    :return: data_matrix: M-by-N numpy array of data values.
    """

    # Check input args.
    check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )

    num_grid_rows = len(nwp_forecast_table_xarray.coords[ROW_DIM].values)
    num_grid_columns = len(nwp_forecast_table_xarray.coords[COLUMN_DIM].values)
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
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=field_name
    )

    nwp_valid_times_unix_sec = (
        init_time_unix_sec +
        nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values *
        HOURS_TO_SECONDS
    )

    bad_hour_flags = numpy.any(
        numpy.isnan(data_matrix_orig_fcst_hours), axis=(1, 2)
    )
    good_hour_indices = numpy.where(numpy.invert(bad_hour_flags))[0]
    data_matrix_orig_fcst_hours = data_matrix_orig_fcst_hours[
        good_hour_indices, ...
    ]
    nwp_valid_times_unix_sec = nwp_valid_times_unix_sec[
        good_hour_indices
    ]

    unique_desired_valid_times_unix_sec = numpy.unique(
        valid_time_matrix_unix_sec
    )

    interp_object = interp1d(
        x=nwp_valid_times_unix_sec,
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
        nwp_forecast_table_xarray, model_name, init_time_unix_sec,
        valid_time_matrix_unix_sec):
    """Reads 24-hour accumulated precip, with different valid time at each px.

    :param nwp_forecast_table_xarray: See documentation for
        `read_nonprecip_field_different_times`.
    :param model_name: Same.
    :param init_time_unix_sec: Same.
    :param valid_time_matrix_unix_sec: Same.
    :return: data_matrix: Same.
    """

    # TODO(thunderhoser): Probably don't need this method.  I needed it for the
    # other fire-Wx project (with Christina) to compute fire-Wx indices -- which
    # are based on weather observations, including past-24-hour precip, at local
    # noon.

    # Check input args.
    check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )

    num_grid_rows = len(nwp_forecast_table_xarray.coords[ROW_DIM].values)
    num_grid_columns = len(nwp_forecast_table_xarray.coords[COLUMN_DIM].values)
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
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=PRECIP_NAME
    )
    assert not numpy.any(numpy.isnan(orig_full_run_precip_matrix_metres))

    orig_forecast_hours = (
        nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values
    )
    all_forecast_hours = model_to_forecast_hours(
        model_name=model_name, init_time_unix_sec=init_time_unix_sec
    )

    if not numpy.all(numpy.isin(all_forecast_hours, orig_forecast_hours)):
        missing_hour_array_string = str(
            all_forecast_hours[
                numpy.isin(all_forecast_hours, orig_forecast_hours) == False
            ]
        )

        warning_string = (
            'POTENTIAL ERROR: Expected {0:d} forecast hours in NWP table.  '
            'Instead, got {1:d} forecast hours, with the following hours '
            'missing:\n{2:s}'
        ).format(
            len(all_forecast_hours),
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
        orig_full_run_precip_matrix_metres = interp_object(all_forecast_hours)
        orig_forecast_hours = all_forecast_hours + 0

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


def precip_from_incremental_to_full_run(nwp_forecast_table_xarray, model_name,
                                        init_time_unix_sec):
    """Converts precip from incremental values to full-run values.

    "Incremental value" = an accumulation between two forecast hours
    "Full-run value" = accumulation over the entire model run, up to a forecast
                       hour

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param model_name: Name of NWP model.
    :param init_time_unix_sec: Initialization time.
    :return: nwp_forecast_table_xarray: Same as input but with full-run precip.
    """

    forecast_hours = nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values
    num_forecast_hours = len(forecast_hours)

    all_forecast_hours = model_to_forecast_hours(
        model_name=model_name, init_time_unix_sec=init_time_unix_sec
    )
    assert numpy.all(numpy.isin(
        element=all_forecast_hours,
        test_elements=forecast_hours
    ))

    data_matrix = nwp_forecast_table_xarray[DATA_KEY].values

    for j in range(num_forecast_hours)[::-1]:
        if model_name in [WRF_ARW_MODEL_NAME, NAM_MODEL_NAME, RAP_MODEL_NAME]:
            addend_indices = numpy.where(forecast_hours <= forecast_hours[j])[0]
        elif model_name == NAM_NEST_MODEL_NAME:
            addend_flags = numpy.logical_or(
                numpy.mod(forecast_hours, 3) == 0,
                forecast_hours == forecast_hours[j]
            )
            addend_flags = numpy.logical_and(
                addend_flags, forecast_hours <= forecast_hours[j]
            )
            addend_indices = numpy.where(addend_flags)[0]
        else:
            addend_indices = None

        for this_field_name in [PRECIP_NAME]:
            k = numpy.where(
                nwp_forecast_table_xarray.coords[FIELD_DIM].values ==
                this_field_name
            )[0][0]

            data_matrix[j, ..., k] = numpy.sum(
                data_matrix[addend_indices, ..., k], axis=0
            )

    nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
        DATA_KEY: (
            nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix
        )
    })

    return nwp_forecast_table_xarray


def remove_negative_precip(nwp_forecast_table_xarray):
    """Removes negative precip accumulations.

    This method requires an input table with full-run values, not incremental
    values -- but it ensures that all incremental values, and thus all full-run
    values, are positive.

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :return: nwp_forecast_table_xarray: Same as input but without negative
        precip accumulations.
    """

    # TODO(thunderhoser): Probably don't need this method.  I needed it for the
    # other fire-Wx project (with Christina), to correct faulty GFS data.

    forecast_hours = nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values
    num_forecast_hours = len(forecast_hours)

    data_matrix = nwp_forecast_table_xarray[DATA_KEY].values

    for j in range(num_forecast_hours):
        if j == 0:
            continue

        for this_field_name in [PRECIP_NAME]:
            ks = numpy.where(
                nwp_forecast_table_xarray.coords[FIELD_DIM].values ==
                this_field_name
            )[0]

            if len(ks) == 0:
                continue

            k = ks[0]
            data_matrix[j, ..., k] = numpy.nanmax(
                data_matrix[:(j + 1), ..., k], axis=0
            )

    nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
        DATA_KEY: (
            nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix
        )
    })

    return nwp_forecast_table_xarray
