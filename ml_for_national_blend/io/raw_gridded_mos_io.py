"""Input/output methods for raw gridded MOS data.

Each raw file should be a TDL Pack file downloaded from the NOAA
High-performance Storage System (HPSS) with the following options:

- One month of model runs at a given initialization hour
  (e.g., all 00Z runs in Feb 2023)
- All forecast hours (valid times)
- Full domain
- Full resolution
- Variables: all keys in dict `FIELD_NAME_TO_TDLPACK_NAME` (defined below)
"""

import os
import numpy
import xarray
import pytdlpack
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.utils import nwp_model_utils

SENTINEL_VALUE = 9999.
VARIABLE_ID_WORD2 = 000000000
MAX_PRECIP_FORECAST_HOUR = 156

DAYS_TO_HOURS = 24
KT_TO_METRES_PER_SECOND = 1.852 / 3.6
INCHES_TO_METRES = 2.54 / 100

FIELD_NAME_TO_TDLPACK_NAME = {
    nwp_model_utils.TEMPERATURE_2METRE_NAME: 222030008,
    nwp_model_utils.DEWPOINT_2METRE_NAME: 223030008,
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: 223075008,
    nwp_model_utils.U_WIND_10METRE_NAME: 224060008,
    nwp_model_utils.V_WIND_10METRE_NAME: 224160008,
    nwp_model_utils.WIND_GUST_10METRE_NAME: 224390008,
    nwp_model_utils.PRECIP_NAME: 223270008
}

FIELD_NAME_TO_CONV_FACTOR = {
    nwp_model_utils.TEMPERATURE_2METRE_NAME:
        temperature_conv.fahrenheit_to_kelvins,
    nwp_model_utils.DEWPOINT_2METRE_NAME:
        temperature_conv.fahrenheit_to_kelvins,
    nwp_model_utils.RELATIVE_HUMIDITY_2METRE_NAME: 0.01,
    nwp_model_utils.U_WIND_10METRE_NAME: KT_TO_METRES_PER_SECOND,
    nwp_model_utils.V_WIND_10METRE_NAME: KT_TO_METRES_PER_SECOND,
    nwp_model_utils.WIND_GUST_10METRE_NAME: KT_TO_METRES_PER_SECOND,
    nwp_model_utils.PRECIP_NAME: INCHES_TO_METRES
}

ALL_FIELD_NAMES = list(FIELD_NAME_TO_TDLPACK_NAME.keys())


def find_file(directory_name, first_init_time_unix_sec,
              raise_error_if_missing=True):
    """Finds TDL Pack file with one month of model runs at a given init hour.

    :param directory_name: Path to input directory.
    :param first_init_time_unix_sec: First init time in month.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: gridded_mos_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    first_init_time_string = time_conversion.unix_sec_to_string(
        first_init_time_unix_sec, '%Y-%m-%d-%H'
    )
    init_year = int(first_init_time_string.split('-')[0])
    init_month = int(first_init_time_string.split('-')[1])
    init_hour = int(first_init_time_string.split('-')[3])

    assert init_hour in [0, 12]

    gridded_mos_file_name = '{0:s}/gfs{1:02d}gmos_co.{2:04d}{3:02d}'.format(
        directory_name, init_hour, init_year, init_month
    )
    if not os.path.isfile(gridded_mos_file_name):
        gridded_mos_file_name = (
            '{0:s}/gfs{1:02d}gmos_co2p5.{2:04d}{3:02d}'
        ).format(
            directory_name, init_hour, init_year, init_month
        )

    if raise_error_if_missing and not os.path.isfile(gridded_mos_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            gridded_mos_file_name
        )
        raise ValueError(error_string)

    return gridded_mos_file_name


def file_name_to_first_init_time(gridded_mos_file_name):
    """Parses first initialization time from name of gridded-MOS file.

    :param gridded_mos_file_name: File path.
    :return: first_init_time_unix_sec: First init time in file.
    """

    error_checking.assert_is_string(gridded_mos_file_name)
    pathless_file_name = os.path.split(gridded_mos_file_name)[1]

    init_hour = int(
        pathless_file_name.replace('gfs', '').replace('gmos', '')
    )
    assert init_hour in [0, 12]

    init_year = int(
        pathless_file_name.split('.')[1][:4]
    )
    init_month = int(
        pathless_file_name.split('.')[1][4:]
    )
    first_init_time_string = '{0:04d}-{1:02d}-01-{2:02d}'.format(
        init_year, init_month, init_hour
    )
    return time_conversion.string_to_unix_sec(
        first_init_time_string, '%Y-%m-%d-%H'
    )


def read_file(tdlpack_file_name, init_time_unix_sec,
              field_names=ALL_FIELD_NAMES):
    """Reads gridded-MOS data from TDL Pack file into xarray table.

    :param tdlpack_file_name: Path to input file.
    :param init_time_unix_sec: Will read this one model run (init time) from the
        file.
    :param field_names: 1-D list with names of fields to read.
    :return: forecast_table_xarray: xarray table with all data.  Metadata
        and variable names should make this table self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_integer(init_time_unix_sec)
    error_checking.assert_is_string(tdlpack_file_name)
    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    latitude_matrix_deg_n, longitude_matrix_deg_e = nbm_utils.read_coords()
    longitude_matrix_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_matrix_deg_e
    )

    forecast_hours = nwp_model_utils.model_to_forecast_hours(
        model_name=nwp_model_utils.GRIDDED_MOS_MODEL_NAME,
        init_time_unix_sec=init_time_unix_sec
    )
    field_names_tdlpack = [FIELD_NAME_TO_TDLPACK_NAME[f] for f in field_names]

    print('Reading data from: "{0:s}"...'.format(tdlpack_file_name))
    tdlpack_file_object = pytdlpack.open(tdlpack_file_name, 'r')
    tdlpack_file_object.rewind()
    print(tdlpack_file_object)

    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]
    num_fields = len(field_names)
    num_forecast_hours = len(forecast_hours)

    data_matrix = numpy.full(
        (num_forecast_hours, num_grid_rows, num_grid_columns, num_fields),
        numpy.nan
    )
    found_data_matrix = numpy.full(
        (num_forecast_hours, num_fields), False, dtype=bool
    )

    while not tdlpack_file_object.eof:
        this_record_object = tdlpack_file_object.read(unpack=True)

        try:
            _ = this_record_object.id
            _ = this_record_object.plain
        except:
            continue

        if this_record_object.id[0] not in field_names_tdlpack:
            continue

        # TODO(thunderhoser): This might change for 12Z model runs.
        if this_record_object.id[1] != VARIABLE_ID_WORD2:
            continue

        this_record_object.unpack(data=True)
        this_data_matrix = numpy.transpose(this_record_object.data)
        this_data_matrix[this_data_matrix >= SENTINEL_VALUE - 1] = numpy.nan

        field_idx = field_names_tdlpack.index(this_record_object.id[0])
        hour_idx = numpy.where(forecast_hours == this_record_object.id[2])[0][0]
        data_matrix[hour_idx, ..., field_idx] = this_data_matrix
        found_data_matrix[hour_idx, field_idx] = True

        print('Found data for field {0:s} at forecast hour {1:d}!'.format(
            field_names[field_idx], forecast_hours[hour_idx]
        ))

    for f in range(num_fields):
        this_conv_factor = FIELD_NAME_TO_CONV_FACTOR[field_names[f]]

        if callable(this_conv_factor):
            data_matrix[..., f] = this_conv_factor(data_matrix[..., f])
        else:
            data_matrix[..., f] *= this_conv_factor

        if field_names[f] == nwp_model_utils.PRECIP_NAME:
            relevant_idxs = numpy.where(
                forecast_hours <= MAX_PRECIP_FORECAST_HOUR
            )[0]
        else:
            relevant_idxs = numpy.linspace(
                0, num_forecast_hours - 1, num=num_forecast_hours, dtype=int
            )

        if numpy.all(found_data_matrix[relevant_idxs, f]):
            continue

        error_string = (
            'Could not find all forecast hours for field {0:s} in file '
            '"{1:s}".  Missing the following hours:\n{2:s}'
        ).format(
            field_names[f],
            tdlpack_file_name,
            str(forecast_hours[found_data_matrix[:, f] == False])
        )

        raise ValueError(error_string)

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM: forecast_hours,
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

    forecast_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    return nwp_model_utils.precip_from_incremental_to_full_run(
        nwp_forecast_table_xarray=forecast_table_xarray,
        model_name=nwp_model_utils.GRIDDED_MOS_MODEL_NAME,
        init_time_unix_sec=init_time_unix_sec
    )
