"""Helper methods for output from any NWP model."""

import time
import warnings
import numpy
import xarray
import pyproj
from scipy.interpolate import interp1d, RegularGridInterpolator
from ml_for_national_blend.outside_code import longitude_conversion as lng_conversion
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.utils import nbm_utils

TOLERANCE = 1e-6

HOURS_TO_SECONDS = 3600
INIT_TIME_FORMAT = '%Y-%m-%d-%H'
DEGREES_TO_RADIANS = numpy.pi / 180.

FORECAST_HOUR_DIM = 'forecast_hour'
ROW_DIM = 'row'
COLUMN_DIM = 'column'
FIELD_DIM = 'field_name'
QUANTILE_LEVEL_DIM = 'quantile_level'
SAMPLE_VALUE_DIM = 'sample'

LATITUDE_KEY = 'latitude_deg_n'
LONGITUDE_KEY = 'longitude_deg_e'
DATA_KEY = 'data_matrix'

MEAN_VALUE_KEY = 'mean_value'
MEAN_SQUARED_VALUE_KEY = 'mean_squared_value'
STDEV_KEY = 'standard_deviation'
QUANTILE_KEY = 'quantile'
PRECIP_SAMPLE_VALUE_KEY = 'precip_sample_value'
NONPRECIP_SAMPLE_VALUE_KEY = 'nonprecip_sample_value'
NUM_VALUES_KEY = 'num_values'

WRF_ARW_MODEL_NAME = 'wrf_arw'
NAM_MODEL_NAME = 'nam'
NAM_NEST_MODEL_NAME = 'nam_nest'
RAP_MODEL_NAME = 'rap'
GFS_MODEL_NAME = 'gfs'
HRRR_MODEL_NAME = 'hrrr'
GEFS_MODEL_NAME = 'gefs'
GRIDDED_LAMP_MODEL_NAME = 'gridded_lamp'
ECMWF_MODEL_NAME = 'ecmwf'
GRIDDED_MOS_MODEL_NAME = 'gridded_gfs_mos'
ENSEMBLE_MODEL_NAME = 'ensemble'

ALL_MODEL_NAMES_SANS_ENSEMBLE = [
    WRF_ARW_MODEL_NAME, NAM_MODEL_NAME, NAM_NEST_MODEL_NAME, RAP_MODEL_NAME,
    GFS_MODEL_NAME, HRRR_MODEL_NAME, GEFS_MODEL_NAME, GRIDDED_LAMP_MODEL_NAME,
    ECMWF_MODEL_NAME, GRIDDED_MOS_MODEL_NAME
]
ALL_MODEL_NAMES_WITH_ENSEMBLE = (
    ALL_MODEL_NAMES_SANS_ENSEMBLE + [ENSEMBLE_MODEL_NAME]
)

MSL_PRESSURE_NAME = 'pressure_mean_sea_level_pascals'
SURFACE_PRESSURE_NAME = 'pressure_surface_pascals'  # ECMWFE control missing this.
TEMPERATURE_2METRE_NAME = 'temperature_2m_agl_kelvins'
DEWPOINT_2METRE_NAME = 'dewpoint_2m_agl_kelvins'
RELATIVE_HUMIDITY_2METRE_NAME = 'relative_humidity_2m_agl'  # ECMWF(D/E) has this at 1000 mb, not 2 m.
U_WIND_10METRE_NAME = 'u_wind_10m_agl_m_s01'
V_WIND_10METRE_NAME = 'v_wind_10m_agl_m_s01'
WIND_GUST_10METRE_NAME = 'wind_gust_10m_agl_m_s01'
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
TEMPERATURE_950MB_NAME = 'temperature_950mb_kelvins'  # ECMWF has this at 925 mb.
MIN_RELATIVE_HUMIDITY_2METRE_NAME = 'hourly_min_relative_humidity_2m_agl'
MAX_RELATIVE_HUMIDITY_2METRE_NAME = 'hourly_max_relative_humidity_2m_agl'

ALL_FIELD_NAMES = [
    MSL_PRESSURE_NAME, SURFACE_PRESSURE_NAME, TEMPERATURE_2METRE_NAME,
    DEWPOINT_2METRE_NAME, RELATIVE_HUMIDITY_2METRE_NAME, U_WIND_10METRE_NAME,
    V_WIND_10METRE_NAME, WIND_GUST_10METRE_NAME, PRECIP_NAME,
    HEIGHT_500MB_NAME, HEIGHT_700MB_NAME,
    RELATIVE_HUMIDITY_500MB_NAME, RELATIVE_HUMIDITY_700MB_NAME,
    RELATIVE_HUMIDITY_850MB_NAME,
    U_WIND_500MB_NAME, U_WIND_700MB_NAME, U_WIND_1000MB_NAME,
    V_WIND_500MB_NAME, V_WIND_700MB_NAME, V_WIND_1000MB_NAME,
    TEMPERATURE_850MB_NAME, TEMPERATURE_950MB_NAME,
    MIN_RELATIVE_HUMIDITY_2METRE_NAME, MAX_RELATIVE_HUMIDITY_2METRE_NAME
]

U_WIND_NAME_TO_V_WIND_NAME = {
    U_WIND_10METRE_NAME: V_WIND_10METRE_NAME,
    U_WIND_500MB_NAME: V_WIND_500MB_NAME,
    U_WIND_700MB_NAME: V_WIND_700MB_NAME,
    U_WIND_1000MB_NAME: V_WIND_1000MB_NAME
}


def _get_rap_wind_rotation_angles(latitude_array_deg_n, longitude_array_deg_e):
    """Computes wind-rotation angle at each RAP pixel.

    I got the projection parameters (standard latitude and central longitude)
    from here:

    https://www.ssec.wisc.edu/realearth/
    solved-north-american-domain-rap-13-5km-rotated-pole-grib2-for-gis/

    And I got the rotation formula from here:

    https://ruc.noaa.gov/ruc/RUC.faq.html

    :param latitude_array_deg_n: numpy array of latitudes (deg north).
    :param longitude_array_deg_e: numpy array of longitudes (deg east) with same
        shape as `latitude_array_deg_n`.
    :return: cosine_array: numpy array with cosines of rotation angles, in same
        shape as `latitude_array_deg_n`.
    :return: sine_array: numpy array with sines of rotation angles, in same
        shape as `latitude_array_deg_n`.
    """

    error_checking.assert_is_valid_lat_numpy_array(
        latitudes_deg=latitude_array_deg_n, allow_nan=False
    )
    longitude_array_pos_in_west_deg_e = (
        lng_conversion.convert_lng_positive_in_west(
            longitudes_deg=longitude_array_deg_e + 0., allow_nan=False
        )
    )
    error_checking.assert_is_numpy_array(
        longitude_array_deg_e,
        exact_dimensions=numpy.array(latitude_array_deg_n.shape, dtype=int)
    )

    standard_latitudes_deg_n = numpy.array([54.])
    central_longitude_deg_e = 254.

    angle_array_radians = (
        numpy.sin(standard_latitudes_deg_n[0] * DEGREES_TO_RADIANS) *
        (longitude_array_pos_in_west_deg_e - central_longitude_deg_e) *
        DEGREES_TO_RADIANS
    )

    return numpy.cos(angle_array_radians), numpy.sin(angle_array_radians)


def check_model_name(model_name, allow_ensemble=False):
    """Ensures that model name is valid.

    :param model_name: String.
    :param allow_ensemble: Boolean flag.  If True, the model name must belong to
        the list `ALL_MODEL_NAMES_WITH_ENSEMBLE`.  If False, the model name must
        belong to the list `ALL_MODEL_NAMES_SANS_ENSEMBLE`.
    :raises: ValueError: if model name does not belong to relevant list.
    """

    error_checking.assert_is_string(model_name)
    error_checking.assert_is_boolean(allow_ensemble)

    if allow_ensemble:
        good_model_names = ALL_MODEL_NAMES_WITH_ENSEMBLE
    else:
        good_model_names = ALL_MODEL_NAMES_SANS_ENSEMBLE

    if model_name in good_model_names:
        return

    error_string = (
        'Model name "{0:s}" is not in the list of accepted model names '
        '(below):\n{1:s}'
    ).format(
        model_name, str(good_model_names)
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

    check_model_name(model_name=model_name, allow_ensemble=True)

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, INIT_TIME_FORMAT
    )
    hour_string = init_time_string.split('-')[-1]

    if model_name in [RAP_MODEL_NAME, HRRR_MODEL_NAME, GRIDDED_LAMP_MODEL_NAME]:
        return

    if model_name in [
            WRF_ARW_MODEL_NAME, ECMWF_MODEL_NAME, GRIDDED_MOS_MODEL_NAME
    ]:
        assert hour_string in ['00', '12']
    else:
        assert hour_string in ['00', '06', '12', '18']


def model_to_init_time_interval(model_name):
    """Returns time interval between successive model runs.

    :param model_name: Name of model.
    :return: init_time_interval_sec: Time interval between successive model
        runs.
    """

    check_model_name(model_name=model_name, allow_ensemble=True)

    # # TODO(thunderhoser): HACK to prevent creation of huge amounts of data.
    # if model_name == HRRR_MODEL_NAME:
    #     return 6 * HOURS_TO_SECONDS

    if model_name in [RAP_MODEL_NAME, HRRR_MODEL_NAME, GRIDDED_LAMP_MODEL_NAME]:
        return HOURS_TO_SECONDS

    if model_name in [
            WRF_ARW_MODEL_NAME, ECMWF_MODEL_NAME, GRIDDED_MOS_MODEL_NAME
    ]:
        return 12 * HOURS_TO_SECONDS

    return 6 * HOURS_TO_SECONDS


def model_to_forecast_hours(model_name, init_time_unix_sec):
    """Returns list of all available forecast hours for the given model.

    :param model_name: Name of model.
    :param init_time_unix_sec: Initialization time.
    :return: all_forecast_hours: 1-D numpy array of integers.
    """

    check_model_name(model_name=model_name, allow_ensemble=True)

    if model_name == RAP_MODEL_NAME:
        init_time_string = time_conversion.unix_sec_to_string(
            init_time_unix_sec, '%Y-%m-%d-%H'
        )
        init_hour_string = init_time_string.split('-')[-1]

        if init_hour_string in ['03', '09', '15', '21']:
            return numpy.linspace(1, 51, num=51, dtype=int)

        return numpy.linspace(1, 21, num=21, dtype=int)

    if model_name == HRRR_MODEL_NAME:
        init_time_string = time_conversion.unix_sec_to_string(
            init_time_unix_sec, '%Y-%m-%d-%H'
        )
        init_hour_string = init_time_string.split('-')[-1]

        if init_hour_string in ['00', '06', '12', '18']:
            return numpy.linspace(1, 48, num=48, dtype=int)

        return numpy.linspace(1, 18, num=18, dtype=int)

    if model_name == NAM_MODEL_NAME:
        return numpy.linspace(51, 84, num=12, dtype=int)

    if model_name == NAM_NEST_MODEL_NAME:
        return numpy.linspace(1, 60, num=60, dtype=int)

    if model_name == GRIDDED_LAMP_MODEL_NAME:
        return numpy.linspace(1, 25, num=25, dtype=int)

    if model_name == GFS_MODEL_NAME:
        return numpy.concatenate([
            numpy.linspace(1, 120, num=120, dtype=int),
            numpy.linspace(123, 384, num=88, dtype=int)
        ])

    if model_name == GEFS_MODEL_NAME:
        return numpy.concatenate([
            numpy.linspace(3, 240, num=80, dtype=int),
            numpy.linspace(246, 384, num=24, dtype=int)
        ])

    if model_name == ECMWF_MODEL_NAME:
        return numpy.linspace(6, 240, num=40, dtype=int)

    if model_name == GRIDDED_MOS_MODEL_NAME:
        return numpy.concatenate([
            numpy.linspace(3, 192, num=64, dtype=int),
            numpy.linspace(198, 264, num=12, dtype=int)
        ])

    if model_name == ENSEMBLE_MODEL_NAME:
        return numpy.linspace(1, 384, num=384, dtype=int)

    return numpy.linspace(1, 48, num=48, dtype=int)


def model_to_old_forecast_hours(model_name):
    """Same as model_to_forecast_hours but for old GFS and GEFS data.

    :param model_name: See doc for `model_to_forecast_hours`.
    :return: all_forecast_hours: Same.
    """

    assert model_name in [GFS_MODEL_NAME, GEFS_MODEL_NAME]

    if model_name == GFS_MODEL_NAME:
        return numpy.concatenate([
            numpy.linspace(3, 240, num=80, dtype=int),
            numpy.linspace(252, 384, num=12, dtype=int)
        ])

    return numpy.linspace(6, 384, num=64, dtype=int)


def model_to_oldish_forecast_hours(model_name):
    """Same as model_to_forecast_hours but for oldish GFS data.

    :param model_name: See doc for `model_to_forecast_hours`.
    :return: all_forecast_hours: Same.
    """

    assert model_name in [GFS_MODEL_NAME]
    return numpy.linspace(3, 384, num=128, dtype=int)


def model_to_maybe_missing_fields(model_name):
    """Returns list of acceptably missing fields for the given model.

    :param model_name: Name of model.
    :return: maybe_missing_field_names: 1-D list with names of acceptably
        missing fields.
    """

    check_model_name(model_name=model_name, allow_ensemble=True)

    if model_name in [
            RAP_MODEL_NAME, GFS_MODEL_NAME, HRRR_MODEL_NAME, GEFS_MODEL_NAME
    ]:
        return [
            WIND_GUST_10METRE_NAME,
            MIN_RELATIVE_HUMIDITY_2METRE_NAME,
            MAX_RELATIVE_HUMIDITY_2METRE_NAME
        ]

    if model_name == GRIDDED_LAMP_MODEL_NAME:
        return [
            MSL_PRESSURE_NAME, SURFACE_PRESSURE_NAME,
            RELATIVE_HUMIDITY_2METRE_NAME, PRECIP_NAME,
            HEIGHT_500MB_NAME, HEIGHT_700MB_NAME,
            RELATIVE_HUMIDITY_500MB_NAME, RELATIVE_HUMIDITY_700MB_NAME,
            RELATIVE_HUMIDITY_850MB_NAME,
            U_WIND_500MB_NAME, U_WIND_700MB_NAME, U_WIND_1000MB_NAME,
            V_WIND_500MB_NAME, V_WIND_700MB_NAME, V_WIND_1000MB_NAME,
            TEMPERATURE_850MB_NAME, TEMPERATURE_950MB_NAME,
            MIN_RELATIVE_HUMIDITY_2METRE_NAME, MAX_RELATIVE_HUMIDITY_2METRE_NAME
        ]

    if model_name == ECMWF_MODEL_NAME:
        return [
            WIND_GUST_10METRE_NAME,
            MIN_RELATIVE_HUMIDITY_2METRE_NAME,
            MAX_RELATIVE_HUMIDITY_2METRE_NAME,
            TEMPERATURE_950MB_NAME
        ]

    if model_name == GRIDDED_MOS_MODEL_NAME:
        return [
            MSL_PRESSURE_NAME, SURFACE_PRESSURE_NAME,
            HEIGHT_500MB_NAME, HEIGHT_700MB_NAME,
            RELATIVE_HUMIDITY_500MB_NAME, RELATIVE_HUMIDITY_700MB_NAME,
            RELATIVE_HUMIDITY_850MB_NAME,
            U_WIND_500MB_NAME, U_WIND_700MB_NAME, U_WIND_1000MB_NAME,
            V_WIND_500MB_NAME, V_WIND_700MB_NAME, V_WIND_1000MB_NAME,
            TEMPERATURE_850MB_NAME, TEMPERATURE_950MB_NAME,
            MIN_RELATIVE_HUMIDITY_2METRE_NAME,
            MAX_RELATIVE_HUMIDITY_2METRE_NAME,
            PRECIP_NAME
        ]

    if model_name == ENSEMBLE_MODEL_NAME:
        return [
            MSL_PRESSURE_NAME, SURFACE_PRESSURE_NAME,
            RELATIVE_HUMIDITY_2METRE_NAME, PRECIP_NAME,
            HEIGHT_500MB_NAME, HEIGHT_700MB_NAME,
            RELATIVE_HUMIDITY_500MB_NAME, RELATIVE_HUMIDITY_700MB_NAME,
            RELATIVE_HUMIDITY_850MB_NAME,
            U_WIND_500MB_NAME, U_WIND_700MB_NAME, U_WIND_1000MB_NAME,
            V_WIND_500MB_NAME, V_WIND_700MB_NAME, V_WIND_1000MB_NAME,
            TEMPERATURE_850MB_NAME, TEMPERATURE_950MB_NAME,
            MIN_RELATIVE_HUMIDITY_2METRE_NAME, MAX_RELATIVE_HUMIDITY_2METRE_NAME
        ]

    return [WIND_GUST_10METRE_NAME]


def model_to_projection(model_name):
    """Returns geographic projection for the given model.

    :param model_name: Name of model.
    :return: proj_object: Instance of `pyproj.Proj`.
    """

    check_model_name(model_name=model_name, allow_ensemble=True)

    if model_name == ENSEMBLE_MODEL_NAME:
        return nbm_utils.NBM_PROJECTION_OBJECT

    if model_name == HRRR_MODEL_NAME:
        return pyproj.Proj(
            proj='lcc', lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=262.5,
            R=6371229., ellps='sphere',
            x_0=2697573.22353293, y_0=1587306.06944136
        )

    if model_name == WRF_ARW_MODEL_NAME:
        return pyproj.Proj(
            proj='lcc', lat_1=25., lat_2=25., lat_0=25., lon_0=265.,
            R=6371229., ellps='sphere',
            x_0=4226106.99691547, y_0=832698.26101756
        )

    if model_name == NAM_MODEL_NAME:
        return pyproj.Proj(
            proj='lcc', lat_1=25., lat_2=25., lat_0=25., lon_0=265.,
            R=6371229., ellps='sphere',
            x_0=4226106.99691547, y_0=832698.26101756
        )

    if model_name == NAM_NEST_MODEL_NAME:
        return pyproj.Proj(
            proj='lcc', lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=262.5,
            R=6371229., ellps='sphere',
            x_0=2697573.22353293, y_0=1587306.06944136
        )

    if model_name == RAP_MODEL_NAME:
        return pyproj.Proj(
            '+proj=ob_tran +o_proj=eqc +o_lon_p=180 +o_lat_p=144 +lon_0=74 '
            '+R=6371229 +x_0=6448701.88 +y_0=5642620.27'
        )

    if model_name == GRIDDED_LAMP_MODEL_NAME:
        return pyproj.Proj(
            proj='lcc', lat_1=25., lat_2=25., lat_0=25., lon_0=265.,
            R=6371229., ellps='sphere',
            x_0=2763216.95215798, y_0=263790.58033545
        )

    if model_name == GRIDDED_MOS_MODEL_NAME:
        return pyproj.Proj(
            proj='lcc', lat_1=25., lat_2=25., lon_0=265.,
            R=6371200., ellps='sphere',
            x_0=3271151.6058371766, y_0=-2604259.810222088
        )

    return None  # Lat-long projection


def model_to_nbm_downsampling_factor(model_name):
    """Returns NBM downsampling factor for the given model.

    "NBM downsampling factor" = how much the National Blend of Models (NBM)
    grid is downsampled when interpolating data from the given model to the NBM
    grid.

    :param model_name: Name of model.
    :return: downsampling_factor: Downsampling factor (positive integer).
    """

    check_model_name(model_name=model_name, allow_ensemble=True)

    if model_name in [RAP_MODEL_NAME, NAM_MODEL_NAME]:
        return 4
    if model_name in [GFS_MODEL_NAME, ECMWF_MODEL_NAME]:
        return 8
    if model_name == GEFS_MODEL_NAME:
        return 16

    return 1


def model_to_nbm_grid_size(model_name):
    """Returns NBM grid size for the given model.

    "NBM grid size" = after interpolating from native NWP grid to NBM grid.
    Some NWP models are interpolated to the full 2.5-km NBM grid, while some are
    interpolated to a downsampled (10-, 20-, or 40-km) version.

    :param model_name: Name of model.
    :return: num_grid_rows: Number of rows in NBM grid.
    :return: num_grid_columns: Number of columns in NBM grid.
    """

    downsampling_factor = model_to_nbm_downsampling_factor(model_name)
    num_grid_rows_full_res = len(nbm_utils.NBM_Y_COORDS_METRES)
    num_grid_columns_full_res = len(nbm_utils.NBM_X_COORDS_METRES)

    if downsampling_factor == 1:
        return num_grid_rows_full_res, num_grid_columns_full_res

    num_grid_rows = int(numpy.floor(
        float(num_grid_rows_full_res) / downsampling_factor
    ))
    num_grid_columns = int(numpy.floor(
        float(num_grid_columns_full_res) / downsampling_factor
    ))

    return num_grid_rows, num_grid_columns


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
        check_model_name(model_name=model_name, allow_ensemble=True)

        if model_name == ENSEMBLE_MODEL_NAME:
            return nbm_utils.read_coords()

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

    try:
        return xarray.concat(
            nwp_forecast_tables_xarray, dim=FORECAST_HOUR_DIM,
            data_vars=[DATA_KEY], coords='minimal', compat='identical',
            join='exact'
        )
    except:
        return xarray.concat(
            nwp_forecast_tables_xarray, dim=FORECAST_HOUR_DIM,
            data_vars=[DATA_KEY], coords='minimal', compat='identical'
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


def subset_by_field(nwp_forecast_table_xarray, desired_field_names):
    """Subsets NWP table by field.

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param desired_field_names: 1-D list with names of desired fields.
    :return: nwp_forecast_table_xarray: Same as input but maybe with fewer
        fields.
    """

    error_checking.assert_is_string_list(desired_field_names)
    return nwp_forecast_table_xarray.sel({FIELD_DIM: desired_field_names})


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


def interp_data_to_nbm_grid(
        nwp_forecast_table_xarray, model_name, use_nearest_neigh,
        interp_to_full_resolution=False, proj_object=None):
    """Interpolates NWP output to the National Blend of Models (NBM) grid.

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param model_name: Model name.
    :param use_nearest_neigh: Boolean flag.  If True (False), will use
        nearest-neighbour (linear) interpolation.
    :param interp_to_full_resolution: Boolean flag.  If True, will interpolate
        to the full-resolution NBM grid, regardless of the source model's
        resolution.  If False, will interpolate to a possibly coarsened version
        of the NBM grid, depending on the source model's resolution.
    :param proj_object: Leave this argument alone.  It's a hack.
    :return: interp_forecast_table_xarray: Same as input but with interpolated
        forecasts.
    """

    check_model_name(model_name=model_name, allow_ensemble=True)
    error_checking.assert_is_boolean(use_nearest_neigh)
    error_checking.assert_is_boolean(interp_to_full_resolution)

    if proj_object is None:
        proj_object = model_to_projection(model_name)

    nwpft = nwp_forecast_table_xarray

    if proj_object is None:
        model_x_matrix_metres = nwpft[LONGITUDE_KEY].values
        model_y_matrix_metres = nwpft[LATITUDE_KEY].values
    else:
        model_x_matrix_metres, model_y_matrix_metres = proj_object(
            nwpft[LONGITUDE_KEY].values, nwpft[LATITUDE_KEY].values
        )

    mean_x_diff_col_to_col = numpy.mean(
        numpy.diff(model_x_matrix_metres, axis=1)
    )
    mean_x_diff_row_to_row = numpy.mean(
        numpy.diff(model_x_matrix_metres, axis=0)
    )
    is_grid_transposed = mean_x_diff_row_to_row > mean_x_diff_col_to_col

    if is_grid_transposed:
        model_x_coords_metres = numpy.mean(model_x_matrix_metres, axis=1)
        model_y_coords_metres = numpy.mean(model_y_matrix_metres, axis=0)
    else:
        model_x_coords_metres = numpy.mean(model_x_matrix_metres, axis=0)
        model_y_coords_metres = numpy.mean(model_y_matrix_metres, axis=1)

    if interp_to_full_resolution:
        downsampling_factor = 1
    else:
        downsampling_factor = model_to_nbm_downsampling_factor(model_name)

    nbm_latitude_matrix_deg_n, nbm_longitude_matrix_deg_e = (
        nbm_utils.read_coords()
    )

    if downsampling_factor > 1:
        dsf = downsampling_factor
        nbm_latitude_matrix_deg_n = (
            nbm_latitude_matrix_deg_n[::dsf, ::dsf][:-1, :-1]
        )
        nbm_longitude_matrix_deg_e = (
            nbm_longitude_matrix_deg_e[::dsf, ::dsf][:-1, :-1]
        )

    if proj_object is None:
        nbm_x_matrix_metres = nbm_longitude_matrix_deg_e
        nbm_y_matrix_metres = nbm_latitude_matrix_deg_n
    else:
        nbm_x_matrix_metres, nbm_y_matrix_metres = proj_object(
            nbm_longitude_matrix_deg_e, nbm_latitude_matrix_deg_n
        )

    nbm_xy_matrix_metres = numpy.transpose(numpy.vstack((
        numpy.ravel(nbm_y_matrix_metres),
        numpy.ravel(nbm_x_matrix_metres)
    )))

    forecast_hours = nwpft.coords[FORECAST_HOUR_DIM].values
    field_names = nwpft.coords[FIELD_DIM].values
    orig_data_matrix = nwpft[DATA_KEY].values

    num_grid_rows_nbm = nbm_x_matrix_metres.shape[0]
    num_grid_columns_nbm = nbm_x_matrix_metres.shape[1]
    num_forecast_hours = len(forecast_hours)
    num_fields = len(field_names)

    these_dim = (
        num_forecast_hours, num_grid_rows_nbm, num_grid_columns_nbm, num_fields
    )
    interp_data_matrix = numpy.full(these_dim, numpy.nan)

    for i in range(num_forecast_hours):
        for j in range(num_fields):
            print('Interpolating {0:s} at forecast hour {1:d}...'.format(
                field_names[j], forecast_hours[i]
            ))

            exec_start_time_unix_sec = time.time()

            interp_object = RegularGridInterpolator(
                points=(model_y_coords_metres, model_x_coords_metres),
                values=(
                    numpy.transpose(orig_data_matrix[i, ..., j])
                    if is_grid_transposed
                    else orig_data_matrix[i, ..., j]
                ),
                method='nearest' if use_nearest_neigh else 'linear',
                bounds_error=False, fill_value=numpy.nan
            )

            these_interp_values = interp_object(nbm_xy_matrix_metres)
            interp_data_matrix[i, ..., j] = numpy.reshape(
                these_interp_values, nbm_x_matrix_metres.shape
            )

            print('Elapsed time = {0:.2f} seconds'.format(
                time.time() - exec_start_time_unix_sec
            ))

    coord_dict = {
        FORECAST_HOUR_DIM: forecast_hours,
        ROW_DIM: numpy.linspace(
            0, num_grid_rows_nbm - 1, num=num_grid_rows_nbm, dtype=int
        ),
        COLUMN_DIM: numpy.linspace(
            0, num_grid_columns_nbm - 1, num=num_grid_columns_nbm, dtype=int
        ),
        FIELD_DIM: field_names
    }

    these_dim = (FORECAST_HOUR_DIM, ROW_DIM, COLUMN_DIM, FIELD_DIM)
    main_data_dict = {
        DATA_KEY: (these_dim, interp_data_matrix)
    }

    these_dim = (ROW_DIM, COLUMN_DIM)
    main_data_dict.update({
        LATITUDE_KEY: (these_dim, nbm_latitude_matrix_deg_n),
        LONGITUDE_KEY: (these_dim, nbm_longitude_matrix_deg_e)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


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
    check_model_name(model_name=model_name, allow_ensemble=False)
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
    check_model_name(model_name=model_name, allow_ensemble=False)
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


def precip_from_incremental_to_full_run(
        nwp_forecast_table_xarray, model_name, init_time_unix_sec,
        be_lenient_with_forecast_hours=False):
    """Converts precip from incremental values to full-run values.

    "Incremental value" = an accumulation between two forecast hours
    "Full-run value" = accumulation over the entire model run, up to a forecast
                       hour

    :param nwp_forecast_table_xarray: xarray table with NWP forecasts.
    :param model_name: Name of NWP model.
    :param init_time_unix_sec: Initialization time.
    :param be_lenient_with_forecast_hours: Boolean flag.
    :return: nwp_forecast_table_xarray: Same as input but with full-run precip.
    """

    check_model_name(model_name=model_name, allow_ensemble=False)
    assert model_name not in [GFS_MODEL_NAME, HRRR_MODEL_NAME, ECMWF_MODEL_NAME]
    error_checking.assert_is_boolean(be_lenient_with_forecast_hours)

    forecast_hours = nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values
    all_forecast_hours = model_to_forecast_hours(
        model_name=model_name, init_time_unix_sec=init_time_unix_sec
    )

    if be_lenient_with_forecast_hours:
        all_forecast_hours = all_forecast_hours[
            all_forecast_hours <= numpy.max(forecast_hours)
        ]

        found_all_hours = numpy.all(numpy.isin(
            element=all_forecast_hours,
            test_elements=forecast_hours
        ))

        if model_name == NAM_NEST_MODEL_NAME and not found_all_hours:
            all_forecast_hours = numpy.concatenate([
                numpy.linspace(1, 24, num=24, dtype=int),
                numpy.linspace(27, 48, num=8, dtype=int)
            ])

    assert numpy.all(numpy.isin(
        element=all_forecast_hours,
        test_elements=forecast_hours
    ))

    data_matrix = nwp_forecast_table_xarray[DATA_KEY].values
    num_forecast_hours = len(forecast_hours)

    for j in range(num_forecast_hours)[::-1]:
        if model_name in [
                WRF_ARW_MODEL_NAME, NAM_MODEL_NAME,
                RAP_MODEL_NAME, GRIDDED_MOS_MODEL_NAME
        ]:
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
        elif model_name == GEFS_MODEL_NAME:
            addend_flags = numpy.logical_or(
                numpy.mod(forecast_hours, 6) == 0,
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

    try:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
            DATA_KEY: (
                nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix
            )
        })
    except:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign(
            DATA_KEY=(nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix)
        )

    return nwp_forecast_table_xarray


def old_gfs_or_gefs_precip_from_incr_to_full(nwp_forecast_table_xarray,
                                             model_name):
    """Same as precip_from_incremental_to_full_run but for old GFS or GEFS data.

    :param nwp_forecast_table_xarray: See doc for
        `precip_from_incremental_to_full_run`.
    :param model_name: Same.
    :return: nwp_forecast_table_xarray: Same.
    """

    assert model_name in [GFS_MODEL_NAME, GEFS_MODEL_NAME]

    forecast_hours = nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values
    all_forecast_hours = model_to_old_forecast_hours(model_name)
    assert numpy.all(numpy.isin(
        element=all_forecast_hours,
        test_elements=forecast_hours
    ))

    data_matrix = nwp_forecast_table_xarray[DATA_KEY].values
    num_forecast_hours = len(forecast_hours)

    for j in range(num_forecast_hours)[::-1]:
        addend_flags = numpy.logical_or(
            numpy.mod(forecast_hours, 6) == 0,
            forecast_hours == forecast_hours[j]
        )
        addend_flags = numpy.logical_and(
            addend_flags, forecast_hours <= forecast_hours[j]
        )
        addend_indices = numpy.where(addend_flags)[0]

        for this_field_name in [PRECIP_NAME]:
            k = numpy.where(
                nwp_forecast_table_xarray.coords[FIELD_DIM].values ==
                this_field_name
            )[0][0]

            data_matrix[j, ..., k] = numpy.sum(
                data_matrix[addend_indices, ..., k], axis=0
            )

    try:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
            DATA_KEY: (
                nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix
            )
        })
    except:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign(
            DATA_KEY=(nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix)
        )

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

    try:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
            DATA_KEY: (
                nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix
            )
        })
    except:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign(
            DATA_KEY=(nwp_forecast_table_xarray[DATA_KEY].dims, data_matrix)
        )

    return nwp_forecast_table_xarray


def rotate_rap_winds_to_earth_relative(nwp_forecast_table_xarray):
    """Rotates RAP winds from grid-relative to Earth-relative.

    :param nwp_forecast_table_xarray: xarray table with RAP forecasts.  Winds
        must be grid-relative (with the u-component running along the rows and
        v-component running along the columns), but this method has no way to
        verify, so BE CAREFUL.
    :return: nwp_forecast_table_xarray: Same but with Earth-relative winds.
    """

    cosine_matrix, sine_matrix = _get_rap_wind_rotation_angles(
        latitude_array_deg_n=nwp_forecast_table_xarray[LATITUDE_KEY].values,
        longitude_array_deg_e=nwp_forecast_table_xarray[LONGITUDE_KEY].values
    )

    forecast_hours = nwp_forecast_table_xarray.coords[FORECAST_HOUR_DIM].values
    num_forecast_hours = len(forecast_hours)

    orig_data_matrix = nwp_forecast_table_xarray[DATA_KEY].values
    new_data_matrix = orig_data_matrix + 0.

    for j in range(num_forecast_hours):
        for this_field_name in list(U_WIND_NAME_TO_V_WIND_NAME.keys()):
            u_index = numpy.where(
                nwp_forecast_table_xarray.coords[FIELD_DIM].values ==
                this_field_name
            )[0][0]

            v_index = numpy.where(
                nwp_forecast_table_xarray.coords[FIELD_DIM].values ==
                U_WIND_NAME_TO_V_WIND_NAME[this_field_name]
            )[0][0]

            new_data_matrix[j, ..., u_index] = (
                cosine_matrix * orig_data_matrix[j, ..., u_index] +
                sine_matrix * orig_data_matrix[j, ..., v_index]
            )

            new_data_matrix[j, ..., v_index] = (
                cosine_matrix * orig_data_matrix[j, ..., v_index] -
                sine_matrix * orig_data_matrix[j, ..., u_index]
            )

    try:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
            DATA_KEY: (
                nwp_forecast_table_xarray[DATA_KEY].dims, new_data_matrix
            )
        })
    except:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign(
            DATA_KEY=(nwp_forecast_table_xarray[DATA_KEY].dims, new_data_matrix)
        )

    return nwp_forecast_table_xarray
