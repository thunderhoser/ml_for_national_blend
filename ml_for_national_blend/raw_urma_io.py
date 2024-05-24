"""Input/output methods for raw URMA data.

URMA = Unrestricted Mesoscale Analysis

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One valid time
- Full domain
- Full resolution
- Variables: all keys in dict `FIELD_NAME_TO_GRIB_NAME` (defined below)
"""

import os
import sys
import warnings
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grib_io
import time_conversion
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import urma_utils

SENTINEL_VALUE = 9.999e20

HOURS_TO_SECONDS = 3600
VALID_TIME_FORMAT_JULIAN = '%y%j%H'

FORMAT_CUTOFF_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2017-05-02-00', '%Y-%m-%d-%H'
)

FIELD_NAME_TO_GRIB_NAME = {
    urma_utils.TEMPERATURE_2METRE_NAME: 'TMP:2 m above ground',
    urma_utils.DEWPOINT_2METRE_NAME: 'DPT:2 m above ground',
    urma_utils.U_WIND_10METRE_NAME: 'UGRD:10 m above ground',
    urma_utils.V_WIND_10METRE_NAME: 'VGRD:10 m above ground',
    urma_utils.WIND_GUST_10METRE_NAME: 'GUST:10 m above ground'
}

ALL_FIELD_NAMES = list(FIELD_NAME_TO_GRIB_NAME.keys())


def find_file(directory_name, valid_time_unix_sec, raise_error_if_missing=True):
    """Finds GRIB2 file with URMA analysis for one time.

    :param directory_name: Path to input directory.
    :param valid_time_unix_sec: Valid time.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: urma_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(valid_time_unix_sec)
    error_checking.assert_is_boolean(raise_error_if_missing)

    valid_date_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, '%Y%m%d'
    )
    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, '%Y-%m-%d-%H'
    )
    hour_string = valid_time_string.split('-')[-1]

    urma_file_name = (
        '{0:s}/{1:s}/urma2p5.t{2:s}z.2dvaranl_ndfd.grb2_wexp'
    ).format(
        directory_name,
        valid_date_string,
        hour_string
    )

    if os.path.isfile(urma_file_name) or not raise_error_if_missing:
        return urma_file_name

    valid_time_string_julian = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, VALID_TIME_FORMAT_JULIAN
    )
    urma_file_name = '{0:s}/{1:s}/{2:s}000000'.format(
        directory_name,
        valid_date_string,
        valid_time_string_julian
    )

    if raise_error_if_missing and not os.path.isfile(urma_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            urma_file_name
        )
        raise ValueError(error_string)

    return urma_file_name


def file_name_to_valid_time(urma_file_name):
    """Parses valid time from name of URMA file.

    :param urma_file_name: File path.
    :return: valid_time_unix_sec: Valid time.
    """

    error_checking.assert_is_string(urma_file_name)

    if urma_file_name.endswith('grb2_wexp'):
        valid_date_string = urma_file_name.split('/')[-2]
        hour_string = urma_file_name.split('/')[-1].split('.')[1]
        hour_string = hour_string.replace('t', '').replace('z', '')
        valid_time_string = '{0:s}{1:s}'.format(valid_date_string, hour_string)

        return time_conversion.string_to_unix_sec(valid_time_string, '%Y%m%d%H')

    pathless_file_name = os.path.split(urma_file_name)[1]
    return time_conversion.string_to_unix_sec(
        pathless_file_name[:7], VALID_TIME_FORMAT_JULIAN
    )


def read_file(grib2_file_name, desired_row_indices, desired_column_indices,
              wgrib2_exe_name, temporary_dir_name, rotate_winds,
              field_names=ALL_FIELD_NAMES):
    """Reads URMA data from GRIB2 file into xarray table.

    :param grib2_file_name: Path to input file.
    :param desired_row_indices: 1-D numpy array with indices of desired grid
        rows.
    :param desired_column_indices: 1-D numpy array with indices of desired grid
        columns.
    :param wgrib2_exe_name: Path to wgrib2 executable.
    :param temporary_dir_name: Path to temporary directory for text files
        created by wgrib2.
    :param rotate_winds: Boolean flag.  If True, will rotate winds from grid-
        relative to Earth-relative.
    :param field_names: 1-D list with names of fields to read.
    :return: urma_table_xarray: xarray table with all data.  Metadata and
        variable names should make this table self-explanatory.
    """

    # Check input args.
    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        urma_utils.read_grid_coords()
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

    error_checking.assert_is_boolean(rotate_winds)
    error_checking.assert_is_string_list(field_names)
    for this_field_name in field_names:
        urma_utils.check_field_name(this_field_name)

    # Do actual stuff.
    valid_time_unix_sec = file_name_to_valid_time(grib2_file_name)

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

    for f in range(num_fields):
        grib_search_string = FIELD_NAME_TO_GRIB_NAME[field_names[f]]

        print('Reading line "{0:s}" from GRIB2 file: "{1:s}"...'.format(
            grib_search_string, grib2_file_name_to_use
        ))
        this_data_matrix = grib_io.read_field_from_grib_file(
            grib_file_name=grib2_file_name_to_use,
            field_name_grib1=grib_search_string,
            num_grid_rows=num_grid_rows,
            num_grid_columns=num_grid_columns,
            wgrib_exe_name=wgrib2_exe_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name,
            sentinel_value=SENTINEL_VALUE,
            raise_error_if_fails=True
            # field_names[f] not in wrf_arw_utils.MAYBE_MISSING_FIELD_NAMES
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

        # TODO(thunderhoser): This is a HACK.  For earlier URMA data (with 200
        # NaN rows), the GRIB2 file is in row-major order -- but for later URMA
        # data (without NaN rows), the GRIB2 file is in column-major order.
        # Fuck literally everything.
        if valid_time_unix_sec >= FORMAT_CUTOFF_TIME_UNIX_SEC:
            orig_dimensions = this_data_matrix.shape
            this_data_matrix = numpy.reshape(
                numpy.ravel(this_data_matrix), orig_dimensions, order='F'
            )
            assert not numpy.any(numpy.isnan(this_data_matrix))

        this_data_matrix = this_data_matrix[desired_row_indices, :]
        this_data_matrix = this_data_matrix[:, desired_column_indices]
        data_matrix[..., f] = this_data_matrix + 0.

    if rotate_winds:
        os.remove(new_grib2_file_name)

    coord_dict = {
        urma_utils.VALID_TIME_DIM:
            numpy.array([valid_time_unix_sec], dtype=int),
        urma_utils.ROW_DIM: desired_row_indices,
        urma_utils.COLUMN_DIM: desired_column_indices,
        urma_utils.FIELD_DIM: field_names
    }

    these_dim = (
        urma_utils.VALID_TIME_DIM, urma_utils.ROW_DIM,
        urma_utils.COLUMN_DIM, urma_utils.FIELD_DIM
    )
    main_data_dict = {
        urma_utils.DATA_KEY: (
            these_dim, numpy.expand_dims(data_matrix, axis=0)
        )
    }

    these_dim = (urma_utils.ROW_DIM, urma_utils.COLUMN_DIM)
    main_data_dict.update({
        urma_utils.LATITUDE_KEY: (these_dim, latitude_matrix_deg_n),
        urma_utils.LONGITUDE_KEY: (these_dim, longitude_matrix_deg_e)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)
