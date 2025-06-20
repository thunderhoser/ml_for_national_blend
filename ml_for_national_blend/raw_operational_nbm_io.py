"""Input/output methods for raw forecast data from operational NBM.

The operational NBM is used as a baseline to compare against our ML models.

Each raw file should be a GRIB2 file downloaded from Amazon Web Services (AWS)
with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Full domain
- Full resolution
- Variables: all keys in dict `FIELD_NAME_TO_GRIB_NAME` (defined below)
"""

import os
import sys
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grib_io
import number_rounding
import time_conversion
import longitude_conversion as lng_conversion
import error_checking
import nbm_utils
import misc_utils
import nwp_model_utils

SENTINEL_VALUE = 9.999e20  # TODO(thunderhoser): Need to check this.

HOURS_TO_SECONDS = 3600
TIME_FORMAT_IN_FILE_NAMES = '%Y%m%d%H'

FIELD_NAME_TO_GRIB_NAME = {
    nwp_model_utils.TEMPERATURE_2METRE_NAME: 'TMP:2 m above ground',
    nwp_model_utils.DEWPOINT_2METRE_NAME: 'DPT:2 m above ground',
    nwp_model_utils.U_WIND_10METRE_NAME: None,
    nwp_model_utils.V_WIND_10METRE_NAME: None,
    nwp_model_utils.WIND_GUST_10METRE_NAME: 'GUST:10 m above ground'
}

FIELD_NAME_TO_CONV_FACTOR = {
    nwp_model_utils.TEMPERATURE_2METRE_NAME: 1.,
    nwp_model_utils.DEWPOINT_2METRE_NAME: 1.,
    nwp_model_utils.U_WIND_10METRE_NAME: 1.,
    nwp_model_utils.V_WIND_10METRE_NAME: 1.,
    nwp_model_utils.WIND_GUST_10METRE_NAME: 1.
}

ALL_FIELD_NAMES = list(FIELD_NAME_TO_GRIB_NAME.keys())


def find_file(directory_name, init_time_unix_sec, forecast_hour,
              raise_error_if_missing=True):
    """Finds GRIB2 file with operational NBM for one init time & one valid time.

    :param directory_name: Path to input directory.
    :param init_time_unix_sec: Initialization time.
    :param forecast_hour: Forecast hour (i.e., lead time) as an integer.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: operational_nbm_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(forecast_hour)
    error_checking.assert_is_greater(forecast_hour, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    rounded_init_time_unix_sec = number_rounding.round_to_nearest(
        init_time_unix_sec, HOURS_TO_SECONDS
    )
    rounded_init_time_unix_sec = rounded_init_time_unix_sec.astype(int)
    error_checking.assert_equals(init_time_unix_sec, rounded_init_time_unix_sec)

    init_time_string = time_conversion.unix_sec_to_string(
        init_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
    )

    operational_nbm_file_name = (
        '{0:s}/{1:s}/blend.t{2:s}z.core.f{3:03d}.co.grib2'
    ).format(
        directory_name,
        init_time_string,
        init_time_string[-2:],
        forecast_hour
    )

    if raise_error_if_missing and not os.path.isfile(operational_nbm_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            operational_nbm_file_name
        )
        raise ValueError(error_string)

    return operational_nbm_file_name


def file_name_to_init_time(operational_nbm_file_name):
    """Parses initialization time from name of operational-NBM file.

    :param operational_nbm_file_name: File path.
    :return: init_time_unix_sec: Initialization time.
    """

    error_checking.assert_is_string(operational_nbm_file_name)
    init_time_string = operational_nbm_file_name.split('/')[-2]
    return time_conversion.string_to_unix_sec(
        init_time_string, TIME_FORMAT_IN_FILE_NAMES
    )


def file_name_to_forecast_hour(operational_nbm_file_name):
    """Parses forecast hour from name of operational-NBM file.

    :param operational_nbm_file_name: File path.
    :return: forecast_hour: Forecast hour (i.e., lead time) as an integer.
    """

    error_checking.assert_is_string(operational_nbm_file_name)
    pathless_file_name = os.path.split(operational_nbm_file_name)[1]
    forecast_hour_part = pathless_file_name.split('.')[3]

    assert forecast_hour_part.startswith('f')
    forecast_hour = int(forecast_hour_part[1:])
    error_checking.assert_is_greater(forecast_hour, 0)

    return forecast_hour


def read_file(
        grib2_file_name, wgrib2_exe_name, temporary_dir_name,
        field_names=ALL_FIELD_NAMES):
    """Reads operational NBM forecasts from GRIB2 file into xarray table.

    :param grib2_file_name: Path to input file.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param temporary_dir_name: Path to temporary directory for text files
        created by wgrib2.
    :param field_names: 1-D list with names of fields to read.
    :return: op_nbm_forecast_table_xarray: xarray table with all data.  Metadata
        and variable names should make this table self-explanatory.
    """

    # Check input args.
    latitude_matrix_deg_n, longitude_matrix_deg_e = nbm_utils.read_coords()
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )

    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        assert this_field_name in ALL_FIELD_NAMES

    # Do actual stuff.
    forecast_hour = file_name_to_forecast_hour(grib2_file_name)

    num_fields = len(field_names)
    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]
    data_matrix = numpy.full(
        (num_grid_rows, num_grid_columns, num_fields), numpy.nan
    )

    grib_inventory_file_name = None

    for f in range(num_fields):
        wind_10m_names = [
            nwp_model_utils.U_WIND_10METRE_NAME,
            nwp_model_utils.V_WIND_10METRE_NAME
        ]

        if field_names[f] in wind_10m_names:
            grib_search_string = 'WIND:10 m above ground'
            print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
                grib_search_string, grib2_file_name
            ))

            (
                speed_matrix_m_s01, grib_inventory_file_name
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
                raise_error_if_fails=True
            )

            grib_search_string = 'WDIR:10 m above ground'
            print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
                grib_search_string, grib2_file_name
            ))

            (
                direction_matrix_deg, grib_inventory_file_name
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

            # TODO(thunderhoser): Who fucking knows?
            # this_data_matrix = numpy.reshape(
            #     numpy.ravel(this_data_matrix), orig_dimensions, order='F'
            # )
            assert not numpy.any(numpy.isnan(this_data_matrix))

            data_matrix[..., f] = this_data_matrix + 0.
            continue

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
            raise_error_if_fails=True
        )

        orig_dimensions = this_data_matrix.shape
        # this_data_matrix = numpy.reshape(
        #     numpy.ravel(this_data_matrix), orig_dimensions, order='F'
        # )
        assert not numpy.any(numpy.isnan(this_data_matrix))

        data_matrix[..., f] = this_data_matrix + 0.

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM:
            numpy.array([forecast_hour], dtype=int),
        nwp_model_utils.ROW_DIM: numpy.linspace(
            0, num_grid_rows - 1, num=num_grid_rows, dtype=int
        ),
        nwp_model_utils.COLUMN_DIM: numpy.linspace(
            0, num_grid_columns - 1, num=num_grid_columns, dtype=int
        ),
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
        nwp_model_utils.LATITUDE_KEY: (these_dim, latitude_matrix_deg_n),
        nwp_model_utils.LONGITUDE_KEY: (these_dim, longitude_matrix_deg_e)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
