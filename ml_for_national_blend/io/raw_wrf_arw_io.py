"""Input/output methods for raw WRF-ARW data.

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
from gewittergefahr.gg_io import grib_io
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.utils import wrf_arw_utils

SENTINEL_VALUE = 9.999e20
DAYS_TO_HOURS = 24

DATE_FORMAT = '%Y%m%d'
INIT_TIME_FORMAT_JULIAN = '%y%j%H'

FIELD_NAME_TO_GRIB_NAME = {
    wrf_arw_utils.MSL_PRESSURE_NAME: 'PRMSL:mean sea level',
    wrf_arw_utils.SURFACE_PRESSURE_NAME: 'PRES:surface',
    wrf_arw_utils.TEMPERATURE_2METRE_NAME: 'TMP:2 m above ground',
    wrf_arw_utils.DEWPOINT_2METRE_NAME: 'DPT:2 m above ground',
    wrf_arw_utils.RELATIVE_HUMIDITY_2METRE_NAME: 'RH:2 m above ground',
    wrf_arw_utils.U_WIND_10METRE_NAME: 'UGRD:10 m above ground',
    wrf_arw_utils.V_WIND_10METRE_NAME: 'VGRD:10 m above ground',
    wrf_arw_utils.PRECIP_NAME: 'APCP:surface',
    wrf_arw_utils.HEIGHT_500MB_NAME: 'HGT:500 mb',
    wrf_arw_utils.HEIGHT_700MB_NAME: 'HGT:700 mb',
    wrf_arw_utils.RELATIVE_HUMIDITY_500MB_NAME: 'RH:500 mb',
    wrf_arw_utils.RELATIVE_HUMIDITY_700MB_NAME: 'RH:700 mb',
    wrf_arw_utils.RELATIVE_HUMIDITY_850MB_NAME: 'RH:850 mb',
    wrf_arw_utils.U_WIND_500MB_NAME: 'UGRD:500 mb',
    wrf_arw_utils.U_WIND_700MB_NAME: 'UGRD:700 mb',
    wrf_arw_utils.U_WIND_1000MB_NAME: 'UGRD:1000 mb',
    wrf_arw_utils.V_WIND_500MB_NAME: 'VGRD:500 mb',
    wrf_arw_utils.V_WIND_700MB_NAME: 'VGRD:700 mb',
    wrf_arw_utils.V_WIND_1000MB_NAME: 'VGRD:1000 mb',
    wrf_arw_utils.TEMPERATURE_850MB_NAME: 'TMP:850 mb',
    wrf_arw_utils.TEMPERATURE_950MB_NAME: 'TMP:950 mb',
    wrf_arw_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: 'MINRH:2 m above ground',
    wrf_arw_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: 'MAXRH:2 m above ground'
}

FIELD_NAME_TO_CONV_FACTOR = {
    wrf_arw_utils.MSL_PRESSURE_NAME: 1.,
    wrf_arw_utils.SURFACE_PRESSURE_NAME: 1.,
    wrf_arw_utils.TEMPERATURE_2METRE_NAME: 1.,
    wrf_arw_utils.DEWPOINT_2METRE_NAME: 1.,
    wrf_arw_utils.RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    wrf_arw_utils.U_WIND_10METRE_NAME: 1.,
    wrf_arw_utils.V_WIND_10METRE_NAME: 1.,
    wrf_arw_utils.PRECIP_NAME: 0.001,
    wrf_arw_utils.HEIGHT_500MB_NAME: 1.,
    wrf_arw_utils.HEIGHT_700MB_NAME: 1.,
    wrf_arw_utils.RELATIVE_HUMIDITY_500MB_NAME: 0.01,
    wrf_arw_utils.RELATIVE_HUMIDITY_700MB_NAME: 0.01,
    wrf_arw_utils.RELATIVE_HUMIDITY_850MB_NAME: 0.01,
    wrf_arw_utils.U_WIND_500MB_NAME: 1.,
    wrf_arw_utils.U_WIND_700MB_NAME: 1.,
    wrf_arw_utils.U_WIND_1000MB_NAME: 1.,
    wrf_arw_utils.V_WIND_500MB_NAME: 1.,
    wrf_arw_utils.V_WIND_700MB_NAME: 1.,
    wrf_arw_utils.V_WIND_1000MB_NAME: 1.,
    wrf_arw_utils.TEMPERATURE_850MB_NAME: 1.,
    wrf_arw_utils.TEMPERATURE_950MB_NAME: 1.,
    wrf_arw_utils.MIN_RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    wrf_arw_utils.MAX_RELATIVE_HUMIDITY_2METRE_NAME: 0.01
}

ALL_FIELD_NAMES = list(FIELD_NAME_TO_GRIB_NAME.keys())


def find_file(directory_name, init_time_unix_sec, forecast_hour,
              raise_error_if_missing=True):
    """Finds GRIB2 file with WRF-ARW data for one init time and one valid time.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.  Must be either 00Z or 12Z.
    :param forecast_hour: Forecast hour (i.e., lead time) as an integer.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: wrf_arw_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    wrf_arw_utils.check_init_time(init_time_unix_sec)
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    init_time_string_julian = time_conversion.unix_sec_to_string(
        init_time_unix_sec, INIT_TIME_FORMAT_JULIAN
    )
    init_date_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, DATE_FORMAT
    )

    wrf_arw_file_name = '{0:s}/{1:s}/{2:s}{3:06d}'.format(
        directory_name,
        init_date_string,
        init_time_string_julian,
        forecast_hour
    )

    if raise_error_if_missing and not os.path.isfile(wrf_arw_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            wrf_arw_file_name
        )
        raise ValueError(error_string)

    return wrf_arw_file_name


def file_name_to_init_time(wrf_arw_file_name):
    """Parses initialization time from name of WRF-ARW file.

    :param wrf_arw_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    error_checking.assert_is_string(wrf_arw_file_name)

    pathless_file_name = os.path.split(wrf_arw_file_name)[1]
    assert len(pathless_file_name) == 13

    init_time_unix_sec = time_conversion.string_to_unix_sec(
        pathless_file_name[:7], INIT_TIME_FORMAT_JULIAN
    )
    wrf_arw_utils.check_init_time(init_time_unix_sec)

    return init_time_unix_sec


def file_name_to_forecast_hour(wrf_arw_file_name):
    """Parses forecast hour from name of WRF-ARW file.

    :param wrf_arw_file_name: File path.
    :return: forecast_hour: Forecast hour (i.e., lead time) as an integer.
    """

    error_checking.assert_is_string(wrf_arw_file_name)

    pathless_file_name = os.path.split(wrf_arw_file_name)[1]
    assert len(pathless_file_name) == 13

    forecast_hour = int(pathless_file_name[-6:])
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)

    return forecast_hour


def read_file(grib2_file_name, desired_row_indices, desired_column_indices,
              read_incremental_precip, wgrib2_exe_name, temporary_dir_name,
              field_names=ALL_FIELD_NAMES):
    """Reads WRF-ARW data from GRIB2 file into xarray table.

    :param grib2_file_name: Path to input file.
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
    :param field_names: 1-D list with names of fields to read.
    :return: wrf_arw_table_xarray: xarray table with all data.  Metadata and
        variable names should make this table self-explanatory.
    """

    # Check input args.
    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        wrf_arw_utils.read_model_coords()
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

    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        wrf_arw_utils.check_field_name(this_field_name)

    # Do actual stuff.
    forecast_hour = file_name_to_forecast_hour(grib2_file_name)

    num_grid_rows = len(desired_row_indices)
    num_grid_columns = len(desired_column_indices)
    num_fields = len(field_names)
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    for f in range(num_fields):
        if field_names[f] in [wrf_arw_utils.PRECIP_NAME]:
            if read_incremental_precip:
                grib_search_string = '{0:s}:{1:d}-{2:d} hour acc'.format(
                    FIELD_NAME_TO_GRIB_NAME[field_names[f]],
                    forecast_hour - 1,
                    forecast_hour
                )
            else:
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
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name,
            field_name_grib1=grib_search_string,
            num_grid_rows=num_grid_rows,
            num_grid_columns=num_grid_columns,
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            raise_error_if_fails=
            field_names[f] not in wrf_arw_utils.MAYBE_MISSING_FIELD_NAMES
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
        assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix[..., f] = (
            this_data_matrix * FIELD_NAME_TO_CONV_FACTOR[field_names[f]]
        )

    coord_dict = {
        wrf_arw_utils.FORECAST_HOUR_DIM:
            numpy.array([forecast_hour], dtype=int),
        wrf_arw_utils.ROW_DIM: desired_row_indices,
        wrf_arw_utils.COLUMN_DIM: desired_column_indices,
        wrf_arw_utils.FIELD_DIM: field_names
    }

    these_dim = (
        wrf_arw_utils.FORECAST_HOUR_DIM, wrf_arw_utils.ROW_DIM,
        wrf_arw_utils.COLUMN_DIM, wrf_arw_utils.FIELD_DIM
    )
    main_data_dict = {
        wrf_arw_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=0)
        )
    }

    these_dim = (wrf_arw_utils.ROW_DIM, wrf_arw_utils.COLUMN_DIM)
    main_data_dict.update({
        wrf_arw_utils.LATITUDE_KEY: (these_dim, latitude_matrix_deg_n),
        wrf_arw_utils.LONGITUDE_KEY: (these_dim, longitude_matrix_deg_e)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
