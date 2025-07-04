"""Input/output methods for raw NWP-forecast data.

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Full domain
- Full resolution
- Variables: all keys in dict `FIELD_NAME_TO_GRIB_NAME` (defined below)
"""

import os
import warnings
import numpy
import xarray
from ml_for_national_blend.outside_code import grib_io
from ml_for_national_blend.outside_code import number_rounding
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import longitude_conversion as lng_conversion
from ml_for_national_blend.outside_code import moisture_conversions as moisture_conv
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.utils import nwp_model_utils

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

SENTINEL_VALUE = 9.999e20
DAYS_TO_HOURS = 24

DATE_FORMAT = '%Y%m%d'
INIT_TIME_FORMAT_JULIAN = '%y%j%H'

FIELD_NAME_TO_GRIB_NAME = {
    nwp_model_utils.MSL_PRESSURE_NAME: 'PRMSL:mean sea level',
    nwp_model_utils.SURFACE_PRESSURE_NAME: 'PRES:surface',
    nwp_model_utils.TEMPERATURE_2METRE_NAME: 'TMP:2 m above ground',
    nwp_model_utils.DEWPOINT_2METRE_NAME: 'DPT:2 m above ground',
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: 'RH:2 m above ground',
    nwp_model_utils.U_WIND_10METRE_NAME: 'UGRD:10 m above ground',
    nwp_model_utils.V_WIND_10METRE_NAME: 'VGRD:10 m above ground',
    nwp_model_utils.WIND_GUST_10METRE_NAME: 'GUST:10 m above ground',
    nwp_model_utils.PRECIP_NAME: 'APCP:surface',
    nwp_model_utils.HEIGHT_500MB_NAME: 'HGT:500 mb',
    nwp_model_utils.HEIGHT_700MB_NAME: 'HGT:700 mb',
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME: 'RH:500 mb',
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME: 'RH:700 mb',
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME: 'RH:850 mb',
    nwp_model_utils.U_WIND_500MB_NAME: 'UGRD:500 mb',
    nwp_model_utils.U_WIND_700MB_NAME: 'UGRD:700 mb',
    nwp_model_utils.U_WIND_1000MB_NAME: 'UGRD:1000 mb',
    nwp_model_utils.V_WIND_500MB_NAME: 'VGRD:500 mb',
    nwp_model_utils.V_WIND_700MB_NAME: 'VGRD:700 mb',
    nwp_model_utils.V_WIND_1000MB_NAME: 'VGRD:1000 mb',
    nwp_model_utils.TEMPERATURE_850MB_NAME: 'TMP:850 mb',
    nwp_model_utils.TEMPERATURE_950MB_NAME: 'TMP:950 mb',
    nwp_model_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: 'MINRH:2 m above ground',
    nwp_model_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: 'MAXRH:2 m above ground'
}

FIELD_NAME_TO_CONV_FACTOR = {
    nwp_model_utils.MSL_PRESSURE_NAME: 1.,
    nwp_model_utils.SURFACE_PRESSURE_NAME: 1.,
    nwp_model_utils.TEMPERATURE_2METRE_NAME: 1.,
    nwp_model_utils.DEWPOINT_2METRE_NAME: 1.,
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    nwp_model_utils.U_WIND_10METRE_NAME: 1.,
    nwp_model_utils.V_WIND_10METRE_NAME: 1.,
    nwp_model_utils.WIND_GUST_10METRE_NAME: 1.,
    nwp_model_utils.PRECIP_NAME: 0.001,
    nwp_model_utils.HEIGHT_500MB_NAME: 1.,
    nwp_model_utils.HEIGHT_700MB_NAME: 1.,
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME: 0.01,
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME: 0.01,
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME: 0.01,
    nwp_model_utils.U_WIND_500MB_NAME: 1.,
    nwp_model_utils.U_WIND_700MB_NAME: 1.,
    nwp_model_utils.U_WIND_1000MB_NAME: 1.,
    nwp_model_utils.V_WIND_500MB_NAME: 1.,
    nwp_model_utils.V_WIND_700MB_NAME: 1.,
    nwp_model_utils.V_WIND_1000MB_NAME: 1.,
    nwp_model_utils.TEMPERATURE_850MB_NAME: 1.,
    nwp_model_utils.TEMPERATURE_950MB_NAME: 1.,
    nwp_model_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    nwp_model_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: 0.01
}

ALL_FIELD_NAMES = list(FIELD_NAME_TO_GRIB_NAME.keys())

FIELD_NAME_TO_GRIB_NAME_ECMWF = {
    nwp_model_utils.MSL_PRESSURE_NAME: 'MSL:sfc',
    nwp_model_utils.SURFACE_PRESSURE_NAME: 'SP:sfc',
    nwp_model_utils.TEMPERATURE_2METRE_NAME: '2T:sfc',
    nwp_model_utils.DEWPOINT_2METRE_NAME: '2D:sfc',
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: '2RH:sfc',  # Will not work.
    nwp_model_utils.U_WIND_10METRE_NAME: '10U:sfc',
    nwp_model_utils.V_WIND_10METRE_NAME: '10V:sfc',
    nwp_model_utils.WIND_GUST_10METRE_NAME: '10GUST:sfc',  # Will not work.
    nwp_model_utils.PRECIP_NAME: 'TP:sfc',
    nwp_model_utils.HEIGHT_500MB_NAME: 'GH:500 mb',
    nwp_model_utils.HEIGHT_700MB_NAME: 'GH:700 mb',
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME: 'R:500 mb',
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME: 'R:700 mb',
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME: 'R:850 mb',
    nwp_model_utils.U_WIND_500MB_NAME: 'U:500 mb',
    nwp_model_utils.U_WIND_700MB_NAME: 'U:700 mb',
    nwp_model_utils.U_WIND_1000MB_NAME: 'U:1000 mb',
    nwp_model_utils.V_WIND_500MB_NAME: 'V:500 mb',
    nwp_model_utils.V_WIND_700MB_NAME: 'V:700 mb',
    nwp_model_utils.V_WIND_1000MB_NAME: 'V:1000 mb',
    nwp_model_utils.TEMPERATURE_850MB_NAME: 'T:850 mb',
    nwp_model_utils.TEMPERATURE_950MB_NAME: 'T:950 mb',  # Will not work.
    nwp_model_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: '2MINRH:sfc',  # Will not work.
    nwp_model_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: '2MAXRH:sfc'  # Will not work.
}

FIELD_NAME_TO_CONV_FACTOR_ECMWF = {
    nwp_model_utils.MSL_PRESSURE_NAME: 1.,
    nwp_model_utils.SURFACE_PRESSURE_NAME: 1.,
    nwp_model_utils.TEMPERATURE_2METRE_NAME: 1.,
    nwp_model_utils.DEWPOINT_2METRE_NAME: 1.,
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    nwp_model_utils.U_WIND_10METRE_NAME: 1.,
    nwp_model_utils.V_WIND_10METRE_NAME: 1.,
    nwp_model_utils.WIND_GUST_10METRE_NAME: 1.,
    nwp_model_utils.PRECIP_NAME: 1.,
    nwp_model_utils.HEIGHT_500MB_NAME: 1.,
    nwp_model_utils.HEIGHT_700MB_NAME: 1.,
    nwp_model_utils.RELATIVE_HUMIDITY_500MB_NAME: 0.01,
    nwp_model_utils.RELATIVE_HUMIDITY_700MB_NAME: 0.01,
    nwp_model_utils.RELATIVE_HUMIDITY_850MB_NAME: 0.01,
    nwp_model_utils.U_WIND_500MB_NAME: 1.,
    nwp_model_utils.U_WIND_700MB_NAME: 1.,
    nwp_model_utils.U_WIND_1000MB_NAME: 1.,
    nwp_model_utils.V_WIND_500MB_NAME: 1.,
    nwp_model_utils.V_WIND_700MB_NAME: 1.,
    nwp_model_utils.V_WIND_1000MB_NAME: 1.,
    nwp_model_utils.TEMPERATURE_850MB_NAME: 1.,
    nwp_model_utils.TEMPERATURE_950MB_NAME: 1.,
    nwp_model_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    nwp_model_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: 0.01
}


def find_file(directory_name, model_name, init_time_unix_sec, forecast_hour,
              raise_error_if_missing=True):
    """Finds GRIB2 file with NWP forecasts for one init time and one valid time.

    :param directory_name: Path to input directory.
    :param model_name: Name of model.
    :param init_time_unix_sec: Initialization time.  Must be either 00Z or 12Z.
    :param forecast_hour: Forecast hour (i.e., lead time) as an integer.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: nwp_forecast_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    init_date_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, DATE_FORMAT
    )
    init_time_string_julian = time_conversion.unix_sec_to_string(
        init_time_unix_sec, INIT_TIME_FORMAT_JULIAN
    )

    if model_name == nwp_model_utils.GRIDDED_LAMP_MODEL_NAME:
        nwp_forecast_file_name = '{0:s}/{1:s}/{2:s}{3:04d}'.format(
            directory_name,
            init_date_string,
            init_time_string_julian,
            forecast_hour
        )

        if not os.path.isfile(nwp_forecast_file_name):
            nwp_forecast_file_name = '{0:s}/{1:s}/{2:s}{3:06d}'.format(
                directory_name,
                init_date_string,
                init_time_string_julian,
                forecast_hour
            )

        if not os.path.isfile(nwp_forecast_file_name):
            nwp_forecast_file_name = '{0:s}/{1:s}/{2:s}{3:05d}'.format(
                directory_name,
                init_date_string,
                init_time_string_julian,
                forecast_hour
            )
    else:
        nwp_forecast_file_name = '{0:s}/{1:s}/{2:s}{3:06d}'.format(
            directory_name,
            init_date_string,
            init_time_string_julian,
            forecast_hour
        )

        if not os.path.isfile(nwp_forecast_file_name):
            nwp_forecast_file_name = '{0:s}/{1:s}/{2:s}{3:05d}'.format(
                directory_name,
                init_date_string,
                init_time_string_julian,
                forecast_hour
            )

        if not os.path.isfile(nwp_forecast_file_name):
            nwp_forecast_file_name = '{0:s}/{1:s}/{2:s}{3:04d}'.format(
                directory_name,
                init_date_string,
                init_time_string_julian,
                forecast_hour
            )

    if raise_error_if_missing and not os.path.isfile(nwp_forecast_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            nwp_forecast_file_name
        )
        raise ValueError(error_string)

    return nwp_forecast_file_name


def file_name_to_init_time(nwp_forecast_file_name, model_name):
    """Parses initialization time from name of NWP-forecast file.

    :param nwp_forecast_file_name: File path.
    :param model_name: Name of model.
    :return: init_time_unix_sec: Initialization time.
    """

    error_checking.assert_is_string(nwp_forecast_file_name)
    pathless_file_name = os.path.split(nwp_forecast_file_name)[1]

    assert len(pathless_file_name) in [11, 12, 13]

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        pathless_file_name[:7], INIT_TIME_FORMAT_JULIAN
    )
    nwp_model_utils.check_init_time(
        init_time_unix_sec=init_time_unix_sec, model_name=model_name
    )
    return init_time_unix_sec


def file_name_to_forecast_hour(nwp_forecast_file_name):
    """Parses forecast hour from name of NWP-forecast file.

    :param nwp_forecast_file_name: File path.
    :return: forecast_hour: Forecast hour (i.e., lead time) as an integer.
    """

    # TODO(thunderhoser): Might want to start verifying forecast hours -- i.e.,
    # that the forecast hour makes sense for the given model -- the same way I
    # already verify init times.

    error_checking.assert_is_string(nwp_forecast_file_name)
    pathless_file_name = os.path.split(nwp_forecast_file_name)[1]

    assert len(pathless_file_name) in [11, 12, 13]

    forecast_hour = int(pathless_file_name[-4:])
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)

    return forecast_hour


def read_file(
        grib2_file_name, model_name,
        desired_row_indices, desired_column_indices,
        read_incremental_precip, wgrib2_exe_name, temporary_dir_name,
        rotate_winds, field_names=ALL_FIELD_NAMES):
    """Reads NWP-forecast data from GRIB2 file into xarray table.

    :param grib2_file_name: Path to input file.
    :param model_name: Name of model.
    :param desired_row_indices: 1-D numpy array with indices of desired grid
        rows.
    :param desired_column_indices: 1-D numpy array with indices of desired grid
        columns.
    :param read_incremental_precip: Boolean flag.  If True, will read
        incremental precip accumulations.  If False, will read precip
        accumulations over the entire model run (i.e., from init time to
        each forecast hour).
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param temporary_dir_name: Path to temporary directory for text files
        created by wgrib2.
    :param rotate_winds: Boolean flag.  If True, will rotate winds from grid-
        relative to Earth-relative.
    :param field_names: 1-D list with names of fields to read.
    :return: nwp_forecast_table_xarray: xarray table with all data.  Metadata
        and variable names should make this table self-explanatory.
    """

    # Check input args.
    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        nwp_model_utils.read_model_coords(model_name=model_name)
    )
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )

    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_column_indices, num_grid_columns
    )

    error_checking.assert_is_numpy_array(
        desired_row_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_row_indices, num_grid_rows
    )
    error_checking.assert_equals_numpy_array(numpy.diff(desired_row_indices), 1)

    error_checking.assert_is_boolean(read_incremental_precip)

    if model_name in [
            nwp_model_utils.GFS_MODEL_NAME, nwp_model_utils.HRRR_MODEL_NAME,
            nwp_model_utils.GRIDDED_LAMP_MODEL_NAME
    ]:
        read_incremental_precip = False

    if model_name in [
            nwp_model_utils.NAM_MODEL_NAME, nwp_model_utils.NAM_NEST_MODEL_NAME,
            nwp_model_utils.GEFS_MODEL_NAME
    ]:
        read_incremental_precip = True

    error_checking.assert_is_boolean(rotate_winds)
    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    forecast_hour = file_name_to_forecast_hour(grib2_file_name)

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_fields = len(field_names)
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    if rotate_winds:
        grid_definition_file_name = '{0:s}/grid_defn.pl'.format(
            THIS_DIRECTORY_NAME
        )
        error_checking.assert_file_exists(grid_definition_file_name)

        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temporary_dir_name
        )
        new_grib2_file_name = '{0:s}/{1:s}'.format(
            temporary_dir_name,
            os.path.split(grib2_file_name)[1]
        )
        grib_io.rotate_winds_in_grib_file(
            input_grib_file_name=grib2_file_name,
            output_grib_file_name=new_grib2_file_name,
            grid_definition_file_name=grid_definition_file_name,
            wgrib2_exe_name=wgrib2_exe_name,
            raise_error_if_fails=True
        )

        grib2_file_name_to_use = new_grib2_file_name
    else:
        new_grib2_file_name = None
        grib2_file_name_to_use = grib2_file_name

    grib_inventory_file_name = None

    for f in range(num_fields):
        wind_10m_names = [
            nwp_model_utils.U_WIND_10METRE_NAME,
            nwp_model_utils.V_WIND_10METRE_NAME
        ]

        if (
                model_name == nwp_model_utils.GRIDDED_LAMP_MODEL_NAME and
                field_names[f] in wind_10m_names
        ):
            grib_search_string = 'WIND:10 m above ground'
            print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
                grib_search_string, grib2_file_name_to_use
            ))

            (
                speed_matrix_m_s01, grib_inventory_file_name
            ) = grib_io.read_field_from_grib_file(
                grib_file_name=grib2_file_name_to_use,
                field_name_grib1=grib_search_string,
                num_grid_rows=latitude_matrix_deg_n.shape[0],
                num_grid_columns=latitude_matrix_deg_n.shape[1],
                wgrib_exe_name=wgrib2_exe_name,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name,
                sentinel_value=SENTINEL_VALUE,
                grib_inventory_file_name=grib_inventory_file_name,
                raise_error_if_fails=True
            )

            grib_search_string = 'WDIR:10 m above ground'
            print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
                grib_search_string, grib2_file_name_to_use
            ))

            (
                direction_matrix_deg, grib_inventory_file_name
            ) = grib_io.read_field_from_grib_file(
                grib_file_name=grib2_file_name_to_use,
                field_name_grib1=grib_search_string,
                num_grid_rows=latitude_matrix_deg_n.shape[0],
                num_grid_columns=latitude_matrix_deg_n.shape[1],
                wgrib_exe_name=wgrib2_exe_name,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name,
                sentinel_value=SENTINEL_VALUE,
                grib_inventory_file_name=grib_inventory_file_name,
                raise_error_if_fails=True
            )

            if field_names[f] == nwp_model_utils.U_WIND_10METRE_NAME:
                this_data_matrix = misc_utils.speed_and_direction_to_uv(
                    wind_speeds_m_s01=speed_matrix_m_s01,
                    wind_directions_deg=direction_matrix_deg
                )[0]
            else:
                this_data_matrix = misc_utils.speed_and_direction_to_uv(
                    wind_speeds_m_s01=speed_matrix_m_s01,
                    wind_directions_deg=direction_matrix_deg
                )[1]

            orig_dimensions = this_data_matrix.shape
            this_data_matrix = numpy.reshape(
                numpy.ravel(this_data_matrix), orig_dimensions, order='F'
            )

            this_data_matrix = this_data_matrix[desired_row_indices, :]
            this_data_matrix = this_data_matrix[:, desired_column_indices]
            # assert not numpy.any(numpy.isnan(this_data_matrix))

            data_matrix[..., f] = this_data_matrix + 0.
            continue

        if field_names[f] in [nwp_model_utils.PRECIP_NAME]:
            if read_incremental_precip:
                if model_name in [
                        nwp_model_utils.WRF_ARW_MODEL_NAME,
                        nwp_model_utils.RAP_MODEL_NAME
                ]:
                    grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        forecast_hour - 1,
                        forecast_hour
                    )
                elif model_name == nwp_model_utils.GEFS_MODEL_NAME:
                    grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        int(number_rounding.floor_to_nearest(
                            forecast_hour - 3, 6
                        )),
                        forecast_hour
                    )
                elif model_name == nwp_model_utils.NAM_MODEL_NAME:
                    grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        forecast_hour - 3,
                        forecast_hour
                    )
                elif model_name == nwp_model_utils.NAM_NEST_MODEL_NAME:
                    grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        int(number_rounding.floor_to_nearest(
                            forecast_hour - 1, 3
                        )),
                        forecast_hour
                    )
                else:
                    grib_search_string = None
            else:
                if (
                        model_name in [
                            nwp_model_utils.WRF_ARW_MODEL_NAME,
                            nwp_model_utils.RAP_MODEL_NAME,
                            nwp_model_utils.GFS_MODEL_NAME,
                            nwp_model_utils.HRRR_MODEL_NAME
                        ] and
                        numpy.mod(forecast_hour, DAYS_TO_HOURS) == 0
                ):
                    grib_search_string = '{0:s}:0-{1:d} day acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        int(numpy.round(float(forecast_hour) / DAYS_TO_HOURS))
                    )
                else:
                    grib_search_string = '{0:s}:0-{1:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        forecast_hour
                    )
        elif (
                model_name == nwp_model_utils.HRRR_MODEL_NAME and
                field_names[f] == nwp_model_utils.MSL_PRESSURE_NAME
        ):
            grib_search_string = 'MSLMA:mean sea level'
        else:
            grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_names[f]]

        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            grib_search_string, grib2_file_name_to_use
        ))

        (
            this_data_matrix, grib_inventory_file_name
        ) = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name_to_use,
            field_name_grib1=grib_search_string,
            num_grid_rows=latitude_matrix_deg_n.shape[0],
            num_grid_columns=latitude_matrix_deg_n.shape[1],
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            grib_inventory_file_name=grib_inventory_file_name,
            raise_error_if_fails=(
                field_names[f] not in
                nwp_model_utils.model_to_maybe_missing_fields(model_name)
            )
        )

        if this_data_matrix is None:
            warning_string = (
                'POTENTIAL ERROR: Cannot find line "{0:s}" in GRIB2 file: '
                '"{1:s}"'
            ).format(
                grib_search_string, grib2_file_name_to_use
            )

            warnings.warn(warning_string)
            continue

        if model_name not in [
                nwp_model_utils.RAP_MODEL_NAME,
                nwp_model_utils.GFS_MODEL_NAME,
                nwp_model_utils.GEFS_MODEL_NAME
        ]:
            orig_dimensions = this_data_matrix.shape
            this_data_matrix = numpy.reshape(
                numpy.ravel(this_data_matrix), orig_dimensions, order='F'
            )

        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        # assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names[f]]
        )

    if rotate_winds:
        os.remove(new_grib2_file_name)

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM:
            numpy.array([forecast_hour], dtype=int),
        nwp_model_utils.ROW_DIM: desired_row_indices,
        nwp_model_utils.COLUMN_DIM: desired_column_indices,
        nwp_model_utils.FIELD_DIM: field_names
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
        nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
    )
    main_data_dict = {
        nwp_model_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=0)
        )
    }

    these_dim = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
    main_data_dict.update({
        nwp_model_utils.LATITUDE_KEY: (
            these_dim,
            latitude_matrix_deg_n[desired_row_indices, :][:, desired_column_indices]
        ),
        nwp_model_utils.LONGITUDE_KEY: (
            these_dim,
            longitude_matrix_deg_e[desired_row_indices, :][:, desired_column_indices]
        )
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def read_ecmwf_file(
        grib_file_name, desired_row_indices, desired_column_indices,
        wgrib_exe_name, temporary_dir_name, field_names=ALL_FIELD_NAMES):
    """Same as read_file but for ECMWF data.

    :param grib_file_name: See documentation for `read_file`.
    :param desired_row_indices: Same.
    :param desired_column_indices: Same.
    :param wgrib_exe_name: Same.
    :param temporary_dir_name: Same.
    :param field_names: Same.
    :return: nwp_forecast_table_xarray: Same.
    """

    # Check input args.
    (
        latitude_matrix_deg_n, longitude_matrix_deg_e
    ) = nwp_model_utils.read_model_coords(
        model_name=nwp_model_utils.ECMWF_MODEL_NAME
    )

    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )

    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_column_indices, num_grid_columns
    )

    error_checking.assert_is_numpy_array(
        desired_row_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_row_indices, num_grid_rows
    )
    error_checking.assert_equals_numpy_array(numpy.diff(desired_row_indices), 1)

    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    forecast_hour = file_name_to_forecast_hour(grib_file_name)

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_fields = len(field_names)
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    # new_grib2_file_name = None
    # grib2_file_name_to_use = grib2_file_name
    grib_inventory_file_name = None

    for f in range(num_fields):
        if field_names[f] == nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME:
            continue

        grib_search_string = FIELD_NAME_TO_GRIB_NAME_ECMWF[field_names[f]]

        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            grib_search_string, grib_file_name
        ))

        (
            this_data_matrix, grib_inventory_file_name
        ) = grib_io.read_field_from_grib_file(
            grib_file_name=grib_file_name,
            field_name_grib1=grib_search_string,
            num_grid_rows=latitude_matrix_deg_n.shape[0],
            num_grid_columns=latitude_matrix_deg_n.shape[1],
            wgrib_exe_name=wgrib_exe_name,
            wgrib2_exe_name=None,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            grib_inventory_file_name=grib_inventory_file_name,
            raise_error_if_fails=(
                field_names[f] not in
                nwp_model_utils.model_to_maybe_missing_fields(
                    nwp_model_utils.ECMWF_MODEL_NAME
                )
            )
        )

        if this_data_matrix is None:
            warning_string = (
                'POTENTIAL ERROR: Cannot find line "{0:s}" in GRIB file: '
                '"{1:s}"'
            ).format(
                grib_search_string, grib_file_name
            )

            warnings.warn(warning_string)
            continue

        this_data_matrix = numpy.flip(this_data_matrix, axis=0)
        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        # assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR_ECMWF[field_names[f]]
        )

    for f in range(num_fields):
        if field_names[f] != nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME:
            continue

        temp_idx = field_names.index(nwp_model_utils.TEMPERATURE_2METRE_NAME)
        dewp_idx = field_names.index(nwp_model_utils.DEWPOINT_2METRE_NAME)
        pres_idx = field_names.index(nwp_model_utils.SURFACE_PRESSURE_NAME)

        data_matrix[..., f] = moisture_conv.dewpoint_to_relative_humidity(
            dewpoints_kelvins=data_matrix[..., dewp_idx],
            temperatures_kelvins=data_matrix[..., temp_idx],
            total_pressures_pascals=data_matrix[..., pres_idx]
        )

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM:
            numpy.array([forecast_hour], dtype=int),
        nwp_model_utils.ROW_DIM: desired_row_indices,
        nwp_model_utils.COLUMN_DIM: desired_column_indices,
        nwp_model_utils.FIELD_DIM: field_names
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
        nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
    )
    main_data_dict = {
        nwp_model_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=0)
        )
    }

    these_dim = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
    main_data_dict.update({
        nwp_model_utils.LATITUDE_KEY: (
            these_dim,
            latitude_matrix_deg_n[desired_row_indices, :][:, desired_column_indices]
        ),
        nwp_model_utils.LONGITUDE_KEY: (
            these_dim,
            longitude_matrix_deg_e[desired_row_indices, :][:, desired_column_indices]
        )
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def read_old_gfs_or_gefs_file(
        grib2_file_name, model_name,
        desired_row_indices, desired_column_indices,
        wgrib2_exe_name, temporary_dir_name,
        field_names=ALL_FIELD_NAMES):
    """Same as read_file but for old GFS or GEFS data.

    Old GFS and GEFS data have fewer lead times and incremental precip, which
    makes things complicated.

    :param grib2_file_name: See documentation for `read_file`.
    :param model_name: Same.
    :param desired_row_indices: Same.
    :param desired_column_indices: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param field_names: Same.
    :return: nwp_forecast_table_xarray: Same.
    """

    # Check input args.
    assert model_name in [
        nwp_model_utils.GFS_MODEL_NAME, nwp_model_utils.GEFS_MODEL_NAME
    ]

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        nwp_model_utils.read_model_coords(model_name=model_name)
    )
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )

    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_column_indices, num_grid_columns
    )

    error_checking.assert_is_numpy_array(
        desired_row_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_row_indices, num_grid_rows
    )
    error_checking.assert_equals_numpy_array(numpy.diff(desired_row_indices), 1)

    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    forecast_hour = file_name_to_forecast_hour(grib2_file_name)

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_fields = len(field_names)
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    grib_inventory_file_name = None

    for f in range(num_fields):
        if field_names[f] in [nwp_model_utils.PRECIP_NAME]:
            if model_name == nwp_model_utils.GFS_MODEL_NAME:
                if forecast_hour <= 240:
                    grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        int(number_rounding.floor_to_nearest(
                            forecast_hour - 3, 6
                        )),
                        forecast_hour
                    )
                else:
                    grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                        FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                        forecast_hour - 12,
                        forecast_hour
                    )
            else:
                grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                    FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                    forecast_hour - 6,
                    forecast_hour
                )
        else:
            grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_names[f]]

        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            grib_search_string, grib2_file_name
        ))

        (
            this_data_matrix, grib_inventory_file_name
        ) = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name,
            field_name_grib1=grib_search_string,
            num_grid_rows=latitude_matrix_deg_n.shape[0],
            num_grid_columns=latitude_matrix_deg_n.shape[1],
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            grib_inventory_file_name=grib_inventory_file_name,
            raise_error_if_fails=(
                field_names[f] not in
                nwp_model_utils.model_to_maybe_missing_fields(model_name)
            )
        )

        if this_data_matrix is None:
            warning_string = (
                'POTENTIAL ERROR: Cannot find line "{0:s}" in GRIB2 file: '
                '"{1:s}"'
            ).format(
                grib_search_string, grib2_file_name
            )

            warnings.warn(warning_string)
            continue

        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        # assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names[f]]
        )

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM:
            numpy.array([forecast_hour], dtype=int),
        nwp_model_utils.ROW_DIM: desired_row_indices,
        nwp_model_utils.COLUMN_DIM: desired_column_indices,
        nwp_model_utils.FIELD_DIM: field_names
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
        nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
    )
    main_data_dict = {
        nwp_model_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=0)
        )
    }

    these_dim = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
    main_data_dict.update({
        nwp_model_utils.LATITUDE_KEY: (
            these_dim,
            latitude_matrix_deg_n[desired_row_indices, :][:, desired_column_indices]
        ),
        nwp_model_utils.LONGITUDE_KEY: (
            these_dim,
            longitude_matrix_deg_e[desired_row_indices, :][:, desired_column_indices]
        )
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def read_oldish_gfs_file(
        grib2_file_name, model_name,
        desired_row_indices, desired_column_indices,
        wgrib2_exe_name, temporary_dir_name,
        field_names=ALL_FIELD_NAMES):
    """Same as read_file but for oldish GFS data.

    Oldish GFS data have fewer lead times.

    :param grib2_file_name: See documentation for `read_file`.
    :param model_name: Same.
    :param desired_row_indices: Same.
    :param desired_column_indices: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param field_names: Same.
    :return: nwp_forecast_table_xarray: Same.
    """

    # Check input args.
    assert model_name in [nwp_model_utils.GFS_MODEL_NAME]

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        nwp_model_utils.read_model_coords(model_name=model_name)
    )
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )

    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    error_checking.assert_is_numpy_array(
        desired_column_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_column_indices)
    error_checking.assert_is_geq_numpy_array(desired_column_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_column_indices, num_grid_columns
    )

    error_checking.assert_is_numpy_array(
        desired_row_indices, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(desired_row_indices)
    error_checking.assert_is_geq_numpy_array(desired_row_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_row_indices, num_grid_rows
    )
    error_checking.assert_equals_numpy_array(numpy.diff(desired_row_indices), 1)

    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    forecast_hour = file_name_to_forecast_hour(grib2_file_name)

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_fields = len(field_names)
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    grib_inventory_file_name = None

    for f in range(num_fields):
        if field_names[f] in [nwp_model_utils.PRECIP_NAME]:
            if numpy.mod(forecast_hour, DAYS_TO_HOURS) == 0:
                grib_search_string = '{0:s}:0-{1:d} day acc'.format(
                    FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                    int(numpy.round(float(forecast_hour) / DAYS_TO_HOURS))
                )
            else:
                grib_search_string = '{0:s}:0-{1:d} hour acc'.format(
                    FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                    forecast_hour
                )
        else:
            grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_names[f]]

        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            grib_search_string, grib2_file_name
        ))

        (
            this_data_matrix, grib_inventory_file_name
        ) = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name,
            field_name_grib1=grib_search_string,
            num_grid_rows=latitude_matrix_deg_n.shape[0],
            num_grid_columns=latitude_matrix_deg_n.shape[1],
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            grib_inventory_file_name=grib_inventory_file_name,
            raise_error_if_fails=(
                field_names[f] not in
                nwp_model_utils.model_to_maybe_missing_fields(model_name)
            )
        )

        if this_data_matrix is None:
            warning_string = (
                'POTENTIAL ERROR: Cannot find line "{0:s}" in GRIB2 file: '
                '"{1:s}"'
            ).format(
                grib_search_string, grib2_file_name
            )

            warnings.warn(warning_string)
            continue

        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        # assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names[f]]
        )

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM:
            numpy.array([forecast_hour], dtype=int),
        nwp_model_utils.ROW_DIM: desired_row_indices,
        nwp_model_utils.COLUMN_DIM: desired_column_indices,
        nwp_model_utils.FIELD_DIM: field_names
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.ROW_DIM,
        nwp_model_utils.COLUMN_DIM, nwp_model_utils.FIELD_DIM
    )
    main_data_dict = {
        nwp_model_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=0)
        )
    }

    these_dim = (nwp_model_utils.ROW_DIM, nwp_model_utils.COLUMN_DIM)
    main_data_dict.update({
        nwp_model_utils.LATITUDE_KEY: (
            these_dim,
            latitude_matrix_deg_n[desired_row_indices, :][:, desired_column_indices]
        ),
        nwp_model_utils.LONGITUDE_KEY: (
            these_dim,
            longitude_matrix_deg_e[desired_row_indices, :][:, desired_column_indices]
        )
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
